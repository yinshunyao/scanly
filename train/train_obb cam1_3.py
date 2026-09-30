#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""相机1/2/3 合并样本 YOLO11-OBB 训练：按配置类别准备四点标签，再调用 Ultralytics OBB。"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_TRAIN_DIR = Path(__file__).resolve().parent
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

from prepare_dataset import prepare_dataset  # noqa: E402
from train import resolve_device, run_train  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    SCANLY_ROOT = Path(__file__).resolve().parents[1]
    TRAIN_DIR = Path(__file__).resolve().parent

    # —— 数据：原始目录与要引入训练的类别（修改后直接运行）——
    SOURCE_DATA_ROOT = SCANLY_ROOT / "样本数据" / "cam123-0820"
    TRAIN_CLASSES = [
        "BandBroken",
        "BandBroken2",
        "Below",
        "Bumps",
        "ClampGlue",
        "GlueGap",
        "GlueResidue",
        "GlueSeam",
        "Hole",
        "Label",
        "Longer",
        "Longer2",
        "ResidualTape",
        "Scrape",
        "Scratch",
        "Shorter",
        "Tackless",
        "Zigzag",
    ]
    OUTPUT_DIR = TRAIN_DIR / "output" / "cam123-0820-obb"
    VAL_RATIO = 0.2
    SEED = 42
    KEEP_EMPTY_LABELS = False
    COPY_IMAGES = True

    # —— 训练 ——
    PREPARE_ONLY = False
    MODEL_PATH = "yolo11s-obb.pt"
    TRAIN_PROJECT = TRAIN_DIR / "runs"
    RUN_NAME = "cam123-0820-obb"
    IMGSZ = 640
    EPOCHS = 100
    BATCH = 8
    PATIENCE = 30
    WORKERS = 4
    DEVICE = ""  # 空则自动选择 cuda / mps / cpu

    stats = prepare_dataset(
        source_root=SOURCE_DATA_ROOT,
        output_dir=OUTPUT_DIR,
        train_classes=TRAIN_CLASSES,
        val_ratio=VAL_RATIO,
        seed=SEED,
        keep_empty_labels=KEEP_EMPTY_LABELS,
        copy_images=COPY_IMAGES,
        label_format="obb",
    )
    logger.info("数据集统计:\n%s", json.dumps(stats, ensure_ascii=False, indent=2))
    if PREPARE_ONLY:
        logger.info("PREPARE_ONLY=True，跳过训练")
        return

    device = DEVICE.strip() or resolve_device()
    run_train(
        data_yaml=OUTPUT_DIR / "data.yaml",
        model_path=MODEL_PATH,
        project=TRAIN_PROJECT,
        run_name=RUN_NAME,
        device=device,
        imgsz=IMGSZ,
        epochs=EPOCHS,
        batch=BATCH,
        patience=PATIENCE,
        workers=WORKERS,
        task="obb",
    )


if __name__ == "__main__":
    main()
