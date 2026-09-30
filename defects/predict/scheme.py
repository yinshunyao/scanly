#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""方案加载、请求覆盖与单框 NG/贴标/排出判定。"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from labels import CATALOG_BY_NAME, cn_name_of

_LOGIC_MAP = {
    "or": "or",
    "或": "or",
    "and": "and",
    "与": "and",
    "length_only": "length_only",
    "length": "length_only",
    "仅长度": "length_only",
}


def normalize_logic(raw: Any) -> str:
    text = str(raw or "or").strip()
    if text in _LOGIC_MAP:
        return _LOGIC_MAP[text]
    lower = text.lower()
    return _LOGIC_MAP.get(lower, "or")


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"配置不是 JSON 对象: {path}")
    return data


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    tmp.write_text(text, encoding="utf-8")
    os.replace(str(tmp), str(dest))


_SCHEME_STRIP_KEYS = {"code", "msg", "engine", "persist", "replace", "persisted"}


def incoming_scheme_overlay(body: dict[str, Any]) -> dict[str, Any]:
    """去掉 GET 回写时的元字段，只保留方案内容。"""
    return {k: v for k, v in body.items() if k not in _SCHEME_STRIP_KEYS}


def has_scheme_fields(overlay: dict[str, Any]) -> bool:
    if overlay.get("scheme_name") not in (None, ""):
        return True
    if overlay.get("mm_per_px") is not None:
        return True
    if isinstance(overlay.get("channels"), dict):
        return True
    if isinstance(overlay.get("defects"), list):
        return True
    return False


def scheme_storage_dict(scheme: dict[str, Any]) -> dict[str, Any]:
    view = scheme_public_view(scheme)
    return {
        "scheme_name": view["scheme_name"],
        "mm_per_px": view["mm_per_px"],
        "channels": view["channels"],
        "defects": view["defects"],
    }


def default_channels() -> dict[str, bool]:
    return {
        "ng": True,
        "ng_area": True,
        "label": True,
        "label_area": False,
        "discharge": True,
        "discharge_area": False,
    }


_CHANNEL_KEYS = (
    "ng",
    "ng_area",
    "label",
    "label_area",
    "discharge",
    "discharge_area",
)


def normalize_channels(raw: Any) -> dict[str, bool]:
    """对齐精度页：NG/贴标/排出各有通道勾选与旁侧「面积」勾选。"""
    out = default_channels()
    if not isinstance(raw, dict):
        return out
    for key in _CHANNEL_KEYS:
        if key in raw:
            out[key] = bool(raw[key])
    if "ng_area" not in raw and "area" in raw:
        out["ng_area"] = bool(raw["area"])
    return out


def _as_rule(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {"height_mm": 0.0, "logic": "or", "length_mm": 0.0}
    return {
        "height_mm": float(raw.get("height_mm") or 0),
        "logic": normalize_logic(raw.get("logic")),
        "length_mm": float(raw.get("length_mm") or 0),
    }


def index_defects(scheme: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in scheme.get("defects") or []:
        if isinstance(item, dict) and item.get("name"):
            out[str(item["name"])] = item
    return out


def merge_scheme(base: dict[str, Any], overlay: dict[str, Any] | None) -> dict[str, Any]:
    """按 name 深合并 defects；未出现的类保持默认。"""
    out = deepcopy(base)
    if not overlay or not isinstance(overlay, dict):
        return out
    if overlay.get("scheme_name"):
        out["scheme_name"] = str(overlay["scheme_name"])
    if isinstance(overlay.get("channels"), dict):
        channels = normalize_channels(out.get("channels"))
        overlay_ch = dict(overlay["channels"])
        channels.update({k: bool(v) for k, v in overlay_ch.items() if k in _CHANNEL_KEYS})
        if "ng_area" not in overlay_ch and "area" in overlay_ch:
            channels["ng_area"] = bool(overlay_ch["area"])
        out["channels"] = normalize_channels(channels)
    if overlay.get("mm_per_px") is not None:
        out["mm_per_px"] = float(overlay["mm_per_px"])
    by_name = index_defects(out)
    for item in overlay.get("defects") or []:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        name = str(item["name"])
        if name in by_name:
            _deep_update(by_name[name], item)
        else:
            out.setdefault("defects", []).append(deepcopy(item))
    return out


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key, value in src.items():
        if key in ("ng", "label", "discharge") and isinstance(value, dict):
            cur = dst.get(key) if isinstance(dst.get(key), dict) else {}
            merged = dict(cur)
            merged.update(value)
            if "logic" in merged:
                merged["logic"] = normalize_logic(merged.get("logic"))
            dst[key] = merged
        else:
            dst[key] = value


def size_rule_hit(height_mm: float, length_mm: float, rule: dict[str, Any] | None) -> bool:
    if not isinstance(rule, dict):
        return False
    logic = normalize_logic(rule.get("logic"))
    h_th = float(rule.get("height_mm") or 0)
    l_th = float(rule.get("length_mm") or 0)
    if logic == "length_only":
        return l_th > 0 and length_mm > l_th
    if h_th <= 0 and l_th <= 0:
        return False
    if logic == "and":
        h_ok = True if h_th <= 0 else height_mm > h_th
        l_ok = True if l_th <= 0 else length_mm > l_th
        return h_ok and l_ok
    parts: list[bool] = []
    if h_th > 0:
        parts.append(height_mm > h_th)
    if l_th > 0:
        parts.append(length_mm > l_th)
    return any(parts) if parts else False


def box_metrics(
    *,
    width_px: float,
    height_px: float,
    mm_per_px: float,
) -> dict[str, float]:
    width_px = max(float(width_px), 0.0)
    height_px = max(float(height_px), 0.0)
    short_px = min(width_px, height_px)
    long_px = max(width_px, height_px)
    scale = float(mm_per_px)
    return {
        "width_px": width_px,
        "height_px": height_px,
        "height_mm": short_px * scale,
        "length_mm": long_px * scale,
        "area_mm2": width_px * height_px * scale * scale,
    }


def bbox_metrics(
    left: float,
    top: float,
    right: float,
    bottom: float,
    mm_per_px: float,
) -> dict[str, float]:
    return box_metrics(
        width_px=max(float(right) - float(left), 0.0),
        height_px=max(float(bottom) - float(top), 0.0),
        mm_per_px=mm_per_px,
    )


def judge_box(
    *,
    name: str,
    score: float,
    left: float,
    top: float,
    right: float,
    bottom: float,
    scheme: dict[str, Any],
    mm_per_px: float,
    laminating: bool,
    width_px: float | None = None,
    height_px: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """
    返回一条 results 项；应丢弃时返回 None。
    未知类：保留但 ng/贴标/排出均为 false。
    width_px/height_px 有值时按该尺寸做 mm 判定（OBB 真实宽高）；否则用外接框。
    """
    channels = normalize_channels(scheme.get("channels"))
    cfg = index_defects(scheme).get(str(name))
    catalog = CATALOG_BY_NAME.get(str(name))
    cn_name = ""
    if cfg and cfg.get("cn_name"):
        cn_name = str(cfg["cn_name"])
    elif catalog:
        cn_name = str(catalog.get("cn_name") or "")
    else:
        cn_name = cn_name_of(name)

    if cfg is not None:
        if cfg.get("enable") is False:
            return None
        if laminating and not bool(cfg.get("detect_when_laminating")):
            return None
        conf_th = float(cfg.get("confidence") if cfg.get("confidence") is not None else 0.0)
        if score < conf_th:
            return None

    if width_px is None or height_px is None:
        metrics = bbox_metrics(left, top, right, bottom, mm_per_px)
    else:
        metrics = box_metrics(width_px=width_px, height_px=height_px, mm_per_px=mm_per_px)
    loc = {
        "left": int(round(left)),
        "top": int(round(top)),
        "width": int(round(max(right - left, 0))),
        "height": int(round(max(bottom - top, 0))),
    }
    item: dict[str, Any] = {
        "name": str(name),
        "cn_name": cn_name,
        "score": round(float(score), 4),
        "location": loc,
        "height_mm": round(metrics["height_mm"], 4),
        "length_mm": round(metrics["length_mm"], 4),
        "area_mm2": round(metrics["area_mm2"], 4),
        "ng": False,
        "need_label": False,
        "need_discharge": False,
    }
    if extra:
        item.update(extra)
    if cfg is None:
        return item

    area_th = float(cfg.get("area_mm2") or 0)
    area_ok = True if area_th <= 0 else metrics["area_mm2"] > area_th

    if bool(channels.get("ng", True)):
        item["ng"] = size_rule_hit(
            metrics["height_mm"], metrics["length_mm"], cfg.get("ng")
        ) and (area_ok if channels.get("ng_area") else True)
    if bool(channels.get("label", True)):
        item["need_label"] = size_rule_hit(
            metrics["height_mm"], metrics["length_mm"], cfg.get("label")
        ) and (area_ok if channels.get("label_area") else True)
    if bool(channels.get("discharge", True)):
        item["need_discharge"] = size_rule_hit(
            metrics["height_mm"], metrics["length_mm"], cfg.get("discharge")
        ) and (area_ok if channels.get("discharge_area") else True)
    return item


def scheme_public_view(scheme: dict[str, Any]) -> dict[str, Any]:
    defects = []
    for item in scheme.get("defects") or []:
        if not isinstance(item, dict):
            continue
        defects.append(
            {
                "name": str(item.get("name") or ""),
                "cn_name": str(item.get("cn_name") or cn_name_of(item.get("name") or "")),
                "enable": bool(item.get("enable", True)),
                "confidence": float(item.get("confidence") or 0),
                "detect_when_laminating": bool(item.get("detect_when_laminating")),
                "area_mm2": float(item.get("area_mm2") or 0),
                "ng": _as_rule(item.get("ng")),
                "label": _as_rule(item.get("label")),
                "discharge": _as_rule(item.get("discharge")),
            }
        )
    return {
        "scheme_name": str(scheme.get("scheme_name") or ""),
        "mm_per_px": float(scheme.get("mm_per_px") or 0.03),
        "channels": normalize_channels(scheme.get("channels")),
        "defects": defects,
    }
