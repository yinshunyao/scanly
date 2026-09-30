#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""封边缺陷中英文对照。name 必须与模型输出 / 客户「AI模型缺陷标签」一致。"""

from __future__ import annotations

from typing import Any

# in_train_set：当前 YOLO 训练 data.yaml 是否含该类
# is_defect：False = 屏蔽非缺陷（ResidualTape/Hole/Zigzag/Label），推理默认不报出
CATALOG: list[dict[str, Any]] = [
    {"name": "GlueSeam", "cn_name": "胶缝", "is_defect": True, "in_train_set": True},
    {"name": "Below", "cn_name": "封边过低", "is_defect": True, "in_train_set": True},
    {"name": "Above", "cn_name": "封边过高", "is_defect": True, "in_train_set": False},
    {"name": "Tackless", "cn_name": "开胶", "is_defect": True, "in_train_set": True},
    {"name": "Shorter", "cn_name": "短带", "is_defect": True, "in_train_set": True},
    {"name": "Longer", "cn_name": "过长", "is_defect": True, "in_train_set": True},
    {"name": "Longer2", "cn_name": "过长2", "is_defect": True, "in_train_set": True},
    {"name": "Scratch", "cn_name": "划伤", "is_defect": True, "in_train_set": True},
    {"name": "Scrape", "cn_name": "刮板", "is_defect": True, "in_train_set": True},
    {"name": "GlueResidue", "cn_name": "残胶", "is_defect": True, "in_train_set": True},
    {"name": "GlueGap", "cn_name": "崩缺", "is_defect": True, "in_train_set": True},
    {"name": "BandBroken", "cn_name": "封边带损伤", "is_defect": True, "in_train_set": True},
    {"name": "BandWave", "cn_name": "波浪纹", "is_defect": True, "in_train_set": False},
    {"name": "Bumps", "cn_name": "鼓包", "is_defect": True, "in_train_set": True},
    {"name": "ClampGlue", "cn_name": "夹胶", "is_defect": True, "in_train_set": True},
    {"name": "ResidualTape", "cn_name": "残带", "is_defect": False, "in_train_set": True},
    {"name": "Hole", "cn_name": "钻孔", "is_defect": False, "in_train_set": True},
    {"name": "Zigzag", "cn_name": "锯齿", "is_defect": False, "in_train_set": True},
    {"name": "Label", "cn_name": "标签", "is_defect": False, "in_train_set": True},
]

CATALOG_BY_NAME: dict[str, dict[str, Any]] = {str(x["name"]): x for x in CATALOG}


def cn_name_of(name: str) -> str:
    item = CATALOG_BY_NAME.get(str(name or ""))
    if not item:
        return ""
    return str(item.get("cn_name") or "")


def labels_payload() -> list[dict[str, Any]]:
    return [
        {
            "name": str(x["name"]),
            "cn_name": str(x["cn_name"]),
            "is_defect": bool(x["is_defect"]),
            "in_train_set": bool(x["in_train_set"]),
        }
        for x in CATALOG
    ]
