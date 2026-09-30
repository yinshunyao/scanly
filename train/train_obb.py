#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""封边缺陷 YOLO11-OBB 训练：按配置类别准备四点标签，再调用 Ultralytics OBB。

数据源读 ``train_detect_cfg``（可被本入口 JSON 覆盖）；超参在 ``train_detect_obb/train_config.json``。
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_TRAIN_DIR = Path(__file__).resolve().parent
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

from train import main  # noqa: E402
from train_detect_cfg.load_cfg import (  # noqa: E402
    load_merged_cfg,
    resolve_source_root,
    split_ratios,
)

logger = logging.getLogger(__name__)

DEFAULT_TRAIN_CONFIG = _TRAIN_DIR / "train_detect_obb" / "train_config.json"


def main_from_config(config_path: str | Path | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg_path = Path(config_path).expanduser().resolve() if config_path else DEFAULT_TRAIN_CONFIG
    cfg, data_cfg_dir, _cfg_dir = load_merged_cfg(cfg_path)
    logger.info("公共配置目录: %s", data_cfg_dir)
    logger.info("入口配置: %s", cfg_path)

    train_cfg = dict(cfg.get("train") or {})
    val_ratio, test_ratio, seed = split_ratios(cfg)
    main(
        source_data_root=resolve_source_root(cfg),
        output_dir=Path(str(cfg["output_dir"])),
        train_classes=list(cfg.get("train_classes") or []),
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        keep_empty_labels=bool(cfg.get("keep_empty_labels", False)),
        copy_images=bool(cfg.get("copy_images", True)),
        skip_prepare=bool(cfg.get("skip_prepare", False)),
        prepare_only=bool(cfg.get("prepare_only", False)),
        model_path=str(cfg.get("model_path") or "yolo11m-obb.pt"),
        train_project=Path(str(cfg.get("train_project") or (_TRAIN_DIR / "runs"))),
        imgsz=int(train_cfg.get("imgsz", 640)),
        epochs=int(train_cfg.get("epochs", 200)),
        batch=int(train_cfg.get("batch", 16)),
        patience=int(train_cfg.get("patience", 30)),
        workers=int(train_cfg.get("workers", 4)),
        device=str(cfg.get("device") or ""),
        single_cls=False,
        label_format="obb",
        task="obb",
    )


if __name__ == "__main__":
    CONFIG_PATH = str(DEFAULT_TRAIN_CONFIG)
    main_from_config(CONFIG_PATH)
