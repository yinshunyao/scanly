#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""单独将检测 core 的 .pth 导出为 ONNX（不跑训练）。

与 ``tools/convert_2_onnx.py``（Ultralytics YOLO .pt → 加密 ONNX）无关。
训练后自动导出仍走 ``train_core.py``；本脚本用于补导出或改 imgsz 后重导。

推理 / ``test_core.py`` 评估 ONNX 时 batch 须为 1（见 ``onnx_export`` 说明）。
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

_TRAIN_DIR = Path(__file__).resolve().parent
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

from train_core import (  # noqa: E402
    DETECT_CORE_ROOT,
    read_class_names,
    resolve_export_checkpoint,
)

logger = logging.getLogger(__name__)


def resolve_checkpoint(raw: str | Path, *, train_dir: Path) -> Path:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("CHECKPOINT 不能为空")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (train_dir / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"未找到 checkpoint: {path}")
    return path


def resolve_output_onnx(
    raw: str | Path | None, *, checkpoint: Path
) -> Path:
    if raw is None or not str(raw).strip():
        return checkpoint.with_suffix(".onnx")
    path = Path(str(raw).strip()).expanduser()
    if not path.is_absolute():
        path = (checkpoint.parent / path).resolve()
    return path


def run_export(
    *,
    checkpoint: Path,
    model_yml: Path,
    output_onnx: Path,
    imgsz: int,
    num_classes: int,
    check: bool = True,
    simplify: bool = False,
    opset_version: int = 16,
) -> Path:
    if not model_yml.is_file():
        raise FileNotFoundError(f"未找到 MODEL_YML: {model_yml}")
    if num_classes < 1:
        raise ValueError(f"num_classes 须 >= 1，当前: {num_classes}")

    prev_cwd = Path.cwd()
    if str(DETECT_CORE_ROOT) not in sys.path:
        sys.path.insert(0, str(DETECT_CORE_ROOT))
    os.chdir(DETECT_CORE_ROOT)
    try:
        from src.core import YAMLConfig
        from src.misc.onnx_export import export_to_onnx

        update: dict[str, Any] = {
            "num_classes": int(num_classes),
            "remap_mscoco_category": False,
            "eval_spatial_size": [int(imgsz), int(imgsz)],
            "PResNet": {"pretrained": False},
            "tuning": None,
        }
        logger.info(
            "导出 ONNX: ckpt=%s out=%s imgsz=%d nc=%d yml=%s",
            checkpoint,
            output_onnx,
            imgsz,
            num_classes,
            model_yml,
        )
        ycfg = YAMLConfig(str(model_yml), **update)
        _ = ycfg.model
        path = export_to_onnx(
            ycfg,
            output_onnx,
            imgsz,
            checkpoint=checkpoint,
            check=check,
            simplify=simplify,
            opset_version=opset_version,
        )
        logger.info("ONNX 已写出: %s（评估请用 test_core.py，batch 强制为 1）", path)
        return path
    finally:
        os.chdir(prev_cwd)


def main(
    *,
    checkpoint: Path,
    model_yml: Path,
    output_onnx: Path | None,
    output_dir: Path | None,
    imgsz: int,
    single_cls: bool,
    num_classes: int | None,
    check: bool,
    simplify: bool,
) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ckpt = resolve_checkpoint(checkpoint, train_dir=_TRAIN_DIR)
    out = resolve_output_onnx(output_onnx, checkpoint=ckpt)

    if num_classes is not None:
        nc = int(num_classes)
    elif single_cls:
        nc = 1
    elif output_dir is not None and output_dir.is_dir():
        nc = len(read_class_names(output_dir))
    else:
        raise ValueError(
            "无法推断 num_classes：请设 NUM_CLASSES，或 SINGLE_CLS=True，"
            "或提供含 classes.txt 的 OUTPUT_DIR"
        )

    return run_export(
        checkpoint=ckpt,
        model_yml=model_yml,
        output_onnx=out,
        imgsz=imgsz,
        num_classes=nc,
        check=check,
        simplify=simplify,
    )


if __name__ == "__main__":
    # IDE / 服务器直接运行：只改下面变量
    # /home/beyond/.conda/envs/yolo11/bin/python3 export_core.py
    TRAIN_DIR = Path(__file__).resolve().parent

    # 例：/data/samples/output/cam1-GlueSeam/core_r50_2/best.pth
    CHECKPOINT = Path("/data/samples/output/cam1-GlueSeam/core_r50_2/best.pth")
    # None=与 checkpoint 同目录同名 .onnx
    OUTPUT_ONNX = None
    # 有 classes.txt 时可填，用于推断多类 nc；SINGLE_CLS=True 时可不填
    OUTPUT_DIR = None
    MODEL_YML = DETECT_CORE_ROOT / "configs" / "core" / "core_r50vd_scanly.yml"
    IMGSZ = 1024
    SINGLE_CLS = True
    NUM_CLASSES = None  # 优先；None 则按 SINGLE_CLS / OUTPUT_DIR 推断
    CHECK = True
    SIMPLIFY = False

    # 若只给了 run 目录，可自动找 best/last：
    ckpt_path = Path(CHECKPOINT)
    if ckpt_path.is_dir():
        found = resolve_export_checkpoint(ckpt_path)
        if found is None:
            raise SystemExit(f"目录内无 best.pth/last.pth: {ckpt_path}")
        ckpt_path = found

    main(
        checkpoint=ckpt_path,
        model_yml=MODEL_YML,
        output_onnx=OUTPUT_ONNX,
        output_dir=OUTPUT_DIR,
        imgsz=IMGSZ,
        single_cls=SINGLE_CLS,
        num_classes=NUM_CLASSES,
        check=CHECK,
        simplify=SIMPLIFY,
    )
