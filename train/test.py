#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""封边缺陷 YOLO11 检测：对已预处理数据集的测试集（默认）做 Ultralytics val。"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

_TRAIN_DIR = Path(__file__).resolve().parent
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

from train_detect_cfg.prepare_dataset import IMAGE_SUFFIXES  # noqa: E402
from train import resolve_device, sync_data_yaml_path  # noqa: E402

logger = logging.getLogger(__name__)


def resolve_model_path(raw: str | Path, *, train_dir: Path) -> Path:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("MODEL_PATH 不能为空")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (train_dir / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"未找到权重: {path}")
    return path


def assert_split_ready(output_dir: Path, data_yaml: Path, split: str) -> None:
    allowed = {"train", "val", "test"}
    if split not in allowed:
        raise ValueError(f"SPLIT 须为 {sorted(allowed)} 之一，当前: {split}")

    has_key = any(
        line.startswith(f"{split}:")
        for line in data_yaml.read_text(encoding="utf-8").splitlines()
    )
    if split == "test" and not has_key:
        raise FileNotFoundError(
            f"data.yaml 缺少 test 字段: {data_yaml}；请设 TEST_RATIO>0 并 "
            "SKIP_PREPARE=False 重新准备数据集"
        )

    image_dir = output_dir / "images" / split
    has_image = image_dir.is_dir() and any(
        p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES for p in image_dir.iterdir()
    )
    if not has_image:
        raise FileNotFoundError(
            f"划分 {split} 图像为空: {image_dir}；请设对应比例后重新准备数据集"
        )


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def metrics_summary(metrics: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "results": _jsonable(getattr(metrics, "results_dict", None) or {}),
    }
    box = getattr(metrics, "box", None)
    if box is not None:
        summary["map50"] = _jsonable(getattr(box, "map50", None))
        summary["map50_95"] = _jsonable(getattr(box, "map", None))
        names = getattr(metrics, "names", None)
        maps = getattr(box, "maps", None)
        if maps is not None and names is not None:
            if isinstance(names, dict):
                name_list = [str(names[i]) for i in sorted(names)]
            else:
                name_list = [str(n) for n in names]
            summary["per_class_map50_95"] = {
                name_list[i]: _jsonable(maps[i])
                for i in range(min(len(name_list), len(maps)))
            }
    save_dir = getattr(metrics, "save_dir", None)
    if save_dir is not None:
        summary["save_dir"] = str(save_dir)
    return summary


def run_val(
    *,
    data_yaml: Path,
    model_path: Path,
    project: Path,
    run_name: str,
    device: str,
    imgsz: int,
    batch: int,
    workers: int,
    split: str,
    single_cls: bool = False,
    extra_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from ultralytics import YOLO

    use_amp = device != "mps"
    kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "split": split,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "workers": workers,
        "project": str(project),
        "name": run_name,
        "amp": use_amp,
        "exist_ok": True,
        "single_cls": single_cls,
        "plots": True,
    }
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    logger.info(
        "开始验证 split=%s yaml=%s model=%s device=%s imgsz=%s batch=%s single_cls=%s",
        split,
        data_yaml,
        model_path,
        device,
        imgsz,
        batch,
        single_cls,
    )
    model = YOLO(str(model_path), task="detect")
    metrics = model.val(**kwargs)
    summary = metrics_summary(metrics)
    out_dir = Path(summary.get("save_dir") or (project / run_name))
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "test_metrics.json"
    metrics_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "验证完成 split=%s mAP50=%s mAP50-95=%s metrics=%s",
        split,
        summary.get("map50"),
        summary.get("map50_95"),
        metrics_path,
    )
    return summary


def main(
    *,
    output_dir: Path,
    model_path: Path,
    train_project: Path,
    run_name: str,
    imgsz: int,
    batch: int,
    workers: int,
    device: str,
    split: str,
    single_cls: bool = False,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    data_yaml = sync_data_yaml_path(output_dir)
    assert_split_ready(output_dir, data_yaml, split)
    resolved_device = device.strip() or resolve_device()
    run_val(
        data_yaml=data_yaml,
        model_path=model_path,
        project=train_project,
        run_name=run_name,
        device=resolved_device,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        split=split,
        single_cls=single_cls,
    )


if __name__ == "__main__":
    # /home/beyond/.conda/envs/yolo11/bin/python3 test.py > test.log 2>&1 &
    TRAIN_DIR = Path(__file__).resolve().parent

    # —— 数据 / 训练对齐 train.py ——
    OUTPUT_DIR = TRAIN_DIR / "output" / "cam-all-0901-train"
    IMGSZ = 1024
    BATCH = 4  # 与 train.py 一致，显存受限勿改
    WORKERS = 2
    DEVICE = ""  # 空则自动选择 cuda / mps / cpu
    SINGLE_CLS = False  # 须与对应训练权重一致
    SPLIT = "test"  # 可改为 val

    # —— 权重与输出 ——
    MODEL_PATH = TRAIN_DIR / "runs" / "cam-all-0901" / "weights" / "best.pt"
    TRAIN_PROJECT = TRAIN_DIR / "runs"
    RUN_NAME = "cam-all-0901-test"

    main(
        output_dir=OUTPUT_DIR,
        model_path=resolve_model_path(MODEL_PATH, train_dir=TRAIN_DIR),
        train_project=TRAIN_PROJECT,
        run_name=RUN_NAME,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        split=SPLIT,
        single_cls=SINGLE_CLS,
    )
