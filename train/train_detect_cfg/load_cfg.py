#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检测训练公共配置：数据根 / 类别 / 划分。

各入口 ``train_config.json`` 写模型与训练超参，经 ``merge_model_cfg`` 叠到本目录 ``data_cfg.json``。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CFG_DIR = Path(__file__).resolve().parent
DATA_CFG_PATH = CFG_DIR / "data_cfg.json"


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"配置不存在: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(raw: str, *, base: Path) -> Path:
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def _absolutize_optional_path(raw: Any, *, base: Path) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    return str(_resolve_path(text, base=base))


def load_data_cfg() -> dict:
    raw = load_json(DATA_CFG_PATH)
    raw["source_data_root"] = _absolutize_optional_path(
        raw.get("source_data_root"), base=CFG_DIR
    )
    raw["voc_xml_path"] = _absolutize_optional_path(raw.get("voc_xml_path"), base=CFG_DIR)
    raw["output_dir"] = _absolutize_optional_path(raw.get("output_dir"), base=CFG_DIR)
    return raw


def merge_model_cfg(
    model_cfg: dict[str, Any] | None,
    *,
    model_cfg_dir: Path,
) -> tuple[dict[str, Any], Path]:
    """公共数据配置为底，入口 JSON 覆盖同名键；``train`` 段逐键覆盖。

    返回 ``(merged, data_cfg_dir)``。未覆盖时数据路径相对本目录；
    入口覆盖后相对 ``model_cfg_dir``。
    """
    shared = load_data_cfg()
    overlay = dict(model_cfg or {})
    out = dict(shared)
    for key, value in overlay.items():
        if key.startswith("_") or key == "train":
            continue
        out[key] = value
    if "source_data_root" in overlay:
        out["source_data_root"] = _absolutize_optional_path(
            overlay.get("source_data_root"), base=model_cfg_dir
        )
    if "voc_xml_path" in overlay:
        out["voc_xml_path"] = _absolutize_optional_path(
            overlay.get("voc_xml_path"), base=model_cfg_dir
        )
    for path_key in ("output_dir", "model_yml", "tuning", "model_path"):
        if path_key in overlay and overlay.get(path_key) not in (None, ""):
            raw = overlay.get(path_key)
            # model_path 可能是 ultralytics 权重名（如 yolo11l.pt），仅相对/绝对路径才解析
            text = str(raw).strip()
            if path_key == "model_path" and "/" not in text and "\\" not in text and not text.endswith(
                (".pt", ".pth", ".yml", ".yaml")
            ):
                out[path_key] = text
            elif path_key == "model_path" and "/" not in text and "\\" not in text and text.endswith(".pt"):
                # 纯文件名预训练权重：保留原样，交给 Ultralytics 下载/查找
                out[path_key] = text
            else:
                out[path_key] = str(_resolve_path(text, base=model_cfg_dir))
    shared_train = dict(shared.get("train") or {})
    overlay_train = dict(overlay.get("train") or {})
    shared_train.update(overlay_train)
    out["train"] = shared_train
    return out, CFG_DIR


def load_merged_cfg(config_path: Path) -> tuple[dict[str, Any], Path, Path]:
    """入口 JSON overlay ``data_cfg.json``。返回 ``(merged, data_cfg_dir, model_cfg_dir)``。"""
    cfg_path = Path(config_path).expanduser().resolve()
    model_cfg = load_json(cfg_path)
    merged, data_cfg_dir = merge_model_cfg(model_cfg, model_cfg_dir=cfg_path.parent)
    return merged, data_cfg_dir, cfg_path.parent


def resolve_source_root(cfg: dict[str, Any]) -> Path:
    """``voc_xml_path`` 非空优先，否则 ``source_data_root``。"""
    voc = cfg.get("voc_xml_path")
    if voc:
        return Path(str(voc))
    src = cfg.get("source_data_root")
    if not src:
        raise ValueError("缺少 source_data_root（且 voc_xml_path 为空）")
    return Path(str(src))


def split_ratios(cfg: dict[str, Any]) -> tuple[float, float, int]:
    split = dict(cfg.get("voc_split") or {})
    return (
        float(split.get("val_ratio", 0.2)),
        float(split.get("test_ratio", 0.0)),
        int(split.get("seed", 42)),
    )
