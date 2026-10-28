#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检测 mark 的 test 画框。不训练，读公共配置里的 val_classes，写到权重所在 run 目录的 vis/，统计写到该目录的 test_summary.json。"""
from __future__ import annotations

import sys
from pathlib import Path

_TRAIN_DIR = Path(__file__).resolve().parent
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

from train_mark import DEFAULT_TRAIN_CONFIG, main_from_config  # noqa: E402


if __name__ == "__main__":
    # 改数据/评估类：train_detect_cfg/data_cfg.json
    # 改权重查找与画框阈值：train_detect_mark/train_config.json
    CONFIG_PATH = str(DEFAULT_TRAIN_CONFIG)
    main_from_config(CONFIG_PATH, test_only=True)
