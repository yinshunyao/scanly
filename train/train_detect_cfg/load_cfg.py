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


def _absolutize_voc_xml_path(raw: Any, *, base: Path) -> str | list[str] | None:
    """字符串、字符串列表，或空。null、空字符串、空列表都视为不用 VOC。"""
    if raw is None:
        return None
    if isinstance(raw, str):
        return _absolutize_optional_path(raw, base=base)
    if isinstance(raw, list):
        paths: list[str] = []
        for item in raw:
            if item is None:
                continue
            text = str(item).strip()
            if not text:
                continue
            paths.append(str(_resolve_path(text, base=base)))
        return paths or None
    raise TypeError(f"voc_xml_path 须为字符串或字符串列表，当前为 {type(raw).__name__}")


def load_data_cfg() -> dict:
    raw = load_json(DATA_CFG_PATH)
    raw["source_data_root"] = _absolutize_optional_path(
        raw.get("source_data_root"), base=CFG_DIR
    )
    raw["voc_xml_path"] = _absolutize_voc_xml_path(raw.get("voc_xml_path"), base=CFG_DIR)
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
        out["voc_xml_path"] = _absolutize_voc_xml_path(
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


def voc_xml_paths(cfg: dict[str, Any]) -> list[Path]:
    """``voc_xml_path`` 规范为路径列表。空则返回 ``[]``。"""
    raw = cfg.get("voc_xml_path")
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        return [Path(text)] if text else []
    if isinstance(raw, list):
        paths: list[Path] = []
        for item in raw:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                paths.append(Path(text))
        return paths
    raise TypeError(f"voc_xml_path 须为字符串或字符串列表，当前为 {type(raw).__name__}")


def resolve_source_root(cfg: dict[str, Any]) -> Path:
    """单个 ``voc_xml_path`` 优先，否则 ``source_data_root``。

    多项 VOC 时调用方应使用 :func:`voc_xml_paths`，不要把列表当成一个根目录。
    """
    vocs = voc_xml_paths(cfg)
    if len(vocs) == 1:
        return vocs[0]
    if len(vocs) > 1:
        raise ValueError("voc_xml_path 含多个目录，请使用 voc_xml_paths()")
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


def resolve_prepare_source(cfg: dict[str, Any]) -> tuple[Path, list[Path] | None]:
    """返回 ``(source_root, voc_paths)``。

    多项 VOC 时 ``voc_paths`` 为全部目录；单项或不用 VOC 时 ``voc_paths`` 为 ``None``。
    """
    vocs = voc_xml_paths(cfg)
    if len(vocs) > 1:
        return vocs[0], vocs
    if len(vocs) == 1:
        return vocs[0], None
    return resolve_source_root(cfg), None


def resolve_val_classes(
    raw: Any,
    class_names: list[str],
    *,
    single_cls: bool = False,
) -> list[str]:
    """选模关注的类别。空列表表示按评估里的全部类别平均。

    非空时须是 ``class_names`` 的子集，且不能与 ``single_cls`` 同时使用：
    单类会把框并成 ``defect``，原类名在评估里已经不存在。
    """
    if raw is None:
        items: list[Any] = []
    elif isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        raise TypeError(
            f"val_classes 须为字符串列表，当前为 {type(raw).__name__}"
        )
    names: list[str] = []
    for item in items:
        name = str(item).strip()
        if not name:
            continue
        if name in names:
            raise ValueError(f"val_classes 重复: {name}")
        names.append(name)
    if not names:
        return []
    if single_cls:
        raise ValueError(
            "single_cls=true 时类别已并成 defect，不能按 val_classes 选模。"
            "请设 single_cls=false，或把 val_classes 置为 []。"
        )
    unknown = [name for name in names if name not in class_names]
    if unknown:
        raise ValueError(
            f"val_classes 不在训练类别中: {unknown}。当前类别: {list(class_names)}"
        )
    return names


def split_each_dir_enabled(cfg: dict[str, Any]) -> bool:
    """多目录是否分目录划分。缺省为打开。"""
    split = dict(cfg.get("voc_split") or {})
    if "split_each_dir" not in split:
        return True
    return bool(split.get("split_each_dir"))
