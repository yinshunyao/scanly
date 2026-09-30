#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""目录批量离线推理：画框图 + Pascal VOC XML。

改下方 ``__main__`` 变量后在 IDE 直接运行。复用 ``predict.json`` / ``scheme.json``，不启 HTTP。
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from predict_all import (
    DEFAULT_PREDICT_JSON,
    DefectPredictAll,
    draw_results,
    load_image_bgr,
)
from scheme import load_json

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(input_dir: Path, *, recursive: bool = False) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    pattern_iter = input_dir.rglob("*") if recursive else input_dir.iterdir()
    out: list[Path] = []
    for path in pattern_iter:
        if not path.is_file():
            continue
        name = path.name
        if name.startswith("._"):
            continue
        if path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        out.append(path)
    out.sort(key=lambda p: p.as_posix().lower())
    return out


def _xml_tostring(root: ET.Element) -> str:
    tree = ET.ElementTree(root)
    try:
        ET.indent(tree, space="    ")
    except AttributeError:
        pass
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(
        root, encoding="unicode"
    ) + "\n"


def result_to_xyxy(item: dict[str, Any]) -> tuple[int, int, int, int] | None:
    loc = item.get("location") or {}
    if isinstance(loc, dict) and loc.get("width") is not None:
        left = int(loc.get("left") or 0)
        top = int(loc.get("top") or 0)
        width = int(loc.get("width") or 0)
        height = int(loc.get("height") or 0)
        if width > 0 and height > 0:
            return left, top, left + width, top + height
    obb = item.get("obb") or {}
    pts = obb.get("points") if isinstance(obb, dict) else None
    if isinstance(pts, list) and len(pts) >= 4:
        xs: list[float] = []
        ys: list[float] = []
        for p in pts[:4]:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                xs.append(float(p[0]))
                ys.append(float(p[1]))
        if xs and ys:
            return (
                int(round(min(xs))),
                int(round(min(ys))),
                int(round(max(xs))),
                int(round(max(ys))),
            )
    return None


def build_voc_xml(
    *,
    image_name: str,
    image_path: Path,
    folder_name: str,
    width: int,
    height: int,
    results: list[dict[str, Any]],
) -> str:
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = folder_name
    ET.SubElement(root, "filename").text = image_name
    ET.SubElement(root, "path").text = str(image_path.resolve())
    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "scanly-defects-predict"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(int(width))
    ET.SubElement(size, "height").text = str(int(height))
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(root, "segmented").text = "0"

    for item in results:
        box = result_to_xyxy(item)
        if box is None:
            continue
        xmin, ymin, xmax, ymax = box
        xmin = max(0, min(width, xmin))
        ymin = max(0, min(height, ymin))
        xmax = max(0, min(width, xmax))
        ymax = max(0, min(height, ymax))
        if xmax <= xmin or ymax <= ymin:
            continue
        name = str(item.get("name") or "unknown").strip() or "unknown"
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = name
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        score = item.get("score")
        if score is not None:
            ET.SubElement(obj, "confidence").text = f"{float(score):.6f}"
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(xmin)
        ET.SubElement(bndbox, "ymin").text = str(ymin)
        ET.SubElement(bndbox, "xmax").text = str(xmax)
        ET.SubElement(bndbox, "ymax").text = str(ymax)
    return _xml_tostring(root)


def imwrite_unicode(path: Path, image_bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".jpg"
    ok, buf = cv2.imencode(suffix, image_bgr)
    if not ok:
        raise RuntimeError(f"编码图像失败: {path}")
    buf.tofile(str(path))


def link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        dst.hardlink_to(src)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def run_dir_predict(
    *,
    input_dir: Path,
    output_dir: Path,
    predict_cfg_path: Path | None = None,
    recursive: bool = False,
    limit: int = 0,
    copy_images: bool = True,
    save_vis: bool = True,
    save_voc: bool = True,
    infer_type: str | None = None,
) -> dict[str, Any]:
    cfg_path = predict_cfg_path or DEFAULT_PREDICT_JSON
    cfg = load_json(cfg_path) if cfg_path.is_file() else {}
    pipeline = DefectPredictAll(predict_cfg=cfg)
    if not pipeline.is_ready(infer_type or pipeline.default_infer_type):
        raise RuntimeError(f"模型未就绪: {pipeline.readiness_payload()}")

    images = list_images(input_dir, recursive=recursive)
    if limit > 0:
        images = images[: int(limit)]
    if not images:
        raise FileNotFoundError(f"输入目录无图像: {input_dir}")

    jpeg_dir = output_dir / "JPEGImages"
    ann_dir = output_dir / "Annotations"
    vis_dir = output_dir / "vis"
    if copy_images:
        jpeg_dir.mkdir(parents=True, exist_ok=True)
    if save_voc:
        ann_dir.mkdir(parents=True, exist_ok=True)
    if save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    class_names: set[str] = set(getattr(pipeline, "class_names", None) or [])
    total_boxes = 0
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []

    for idx, src in enumerate(images, start=1):
        t_img = time.perf_counter()
        image_bgr = load_image_bgr(src)
        payload = pipeline.predict_bgr(
            image_bgr,
            image_path=str(src),
            infer_type=infer_type,
        )
        results = list(payload.get("results") or [])
        h, w = image_bgr.shape[:2]
        for item in results:
            name = str(item.get("name") or "").strip()
            if name:
                class_names.add(name)
        total_boxes += len(results)

        jpeg_path = jpeg_dir / src.name if copy_images else src
        if copy_images:
            link_or_copy(src, jpeg_path)

        if save_voc:
            xml_text = build_voc_xml(
                image_name=src.name,
                image_path=jpeg_path,
                folder_name="JPEGImages" if copy_images else src.parent.name,
                width=w,
                height=h,
                results=results,
            )
            (ann_dir / f"{src.stem}.xml").write_text(xml_text, encoding="utf-8")

        if save_vis:
            canvas = draw_results(image_bgr, results)
            imwrite_unicode(vis_dir / src.name, canvas)

        elapsed = time.perf_counter() - t_img
        rows.append(
            {
                "image": src.name,
                "boxes": len(results),
                "seconds": round(elapsed, 3),
            }
        )
        logger.info(
            "[%d/%d] %s boxes=%d %.2fs",
            idx,
            len(images),
            src.name,
            len(results),
            elapsed,
        )

    names_sorted = sorted(class_names) if class_names else ["defect"]
    (output_dir / "predefined_classes.txt").write_text(
        "\n".join(names_sorted) + "\n", encoding="utf-8"
    )
    summary = {
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "infer_type": infer_type or pipeline.default_infer_type,
        "model_path": pipeline.model_path,
        "obb_model_path": pipeline.obb_model_path,
        "image_count": len(images),
        "box_count": total_boxes,
        "class_names": names_sorted,
        "elapsed_sec": round(time.perf_counter() - t0, 3),
        "copy_images": bool(copy_images),
        "save_vis": bool(save_vis),
        "save_voc": bool(save_voc),
        "images": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "完成: images=%d boxes=%d elapsed=%.1fs -> %s",
        len(images),
        total_boxes,
        summary["elapsed_sec"],
        output_dir,
    )
    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # —— IDE 直接改这里 ——
    INPUT_DIR = Path("/Volumes/shunyao-h1/scanly/测试数据/cam2-test")
    OUTPUT_DIR = Path("/Volumes/shunyao-h1/scanly/测试数据/cam2-test-pred-v2.3-c2")
    PREDICT_JSON = DEFAULT_PREDICT_JSON  # 或自定义 predict.json
    RECURSIVE = False
    LIMIT = 0  # >0 只跑前 N 张；全量保持 0
    COPY_IMAGES = True
    SAVE_VIS = True
    SAVE_VOC = True
    INFER_TYPE = None  # None=用 predict.json；可强制 "detect" / "obb"

    run_dir_predict(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        predict_cfg_path=PREDICT_JSON,
        recursive=RECURSIVE,
        limit=LIMIT,
        copy_images=COPY_IMAGES,
        save_vis=SAVE_VIS,
        save_voc=SAVE_VOC,
        infer_type=INFER_TYPE,
    )
