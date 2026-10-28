#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""封边缺陷 YOLO11 检测训练：先按配置类别准备数据，再调用 Ultralytics。

数据源读 ``train_detect_cfg``；训练超参在 ``train_detect_yolo/train_config.json``。
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

_TRAIN_DIR = Path(__file__).resolve().parent
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

from train_detect_cfg.prepare_dataset import prepare_dataset  # noqa: E402
from train_detect_cfg.load_cfg import (  # noqa: E402
    load_merged_cfg,
    resolve_prepare_source,
    split_each_dir_enabled,
    split_ratios,
)

logger = logging.getLogger(__name__)

DEFAULT_TRAIN_CONFIG = _TRAIN_DIR / "train_detect_yolo" / "train_config.json"


def resolve_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if sys.platform == "linux" and torch.cuda.is_available():
        return "cuda:0"
    if sys.platform == "darwin" and hasattr(torch, "mps") and torch.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def assert_prepared_dataset(output_dir: Path) -> Path:
    data_yaml = output_dir / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"未找到已预处理数据集: {data_yaml}；请设 skip_prepare=false 重新准备，"
            "或运行 train_detect_cfg/prepare_dataset.py"
        )
    return data_yaml


def sync_data_yaml_path(output_dir: Path) -> Path:
    """将 data.yaml 的 path 同步为当前机器的 output_dir（跨机器/目录训练）。"""
    data_yaml = assert_prepared_dataset(output_dir)
    resolved = output_dir.resolve()
    lines: list[str] = []
    path_updated = False
    for raw in data_yaml.read_text(encoding="utf-8").splitlines():
        if raw.startswith("path:"):
            lines.append(f"path: {resolved}")
            path_updated = True
        else:
            lines.append(raw)
    if not path_updated:
        raise ValueError(f"data.yaml 缺少 path 字段: {data_yaml}")
    if lines and lines[-1]:
        lines.append("")
    data_yaml.write_text("\n".join(lines), encoding="utf-8")
    logger.info("已同步 data.yaml path -> %s", resolved)
    return data_yaml


# Ultralytics detect 官方默认增强（自动增强）。train 段可覆盖任一键。
_DEFAULT_AUGMENT: dict[str, Any] = {
    "augment": True,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "fliplr": 0.5,
    "flipud": 0.0,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "erasing": 0.4,
    "close_mosaic": 10,
    "multi_scale": False,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "max_det": 300,
}


def build_augment_train_kwargs(train_cfg: dict[str, Any]) -> dict[str, Any]:
    """从 train 配置组装增强 kwargs；缺省用 Ultralytics 自动增强默认。"""
    out: dict[str, Any] = {}
    for key, default in _DEFAULT_AUGMENT.items():
        if key not in train_cfg and key != "augment":
            # 未写增强键时：仅当 augment 显式/默认为 true 才填官方默认
            continue
        raw = train_cfg.get(key, default)
        if key in {"close_mosaic", "max_det"}:
            out[key] = int(raw)
        elif key in {"augment", "multi_scale"}:
            out[key] = bool(raw)
        else:
            out[key] = float(raw)
    if "augment" not in out:
        out["augment"] = bool(train_cfg.get("augment", True))
    if out["augment"]:
        for key, default in _DEFAULT_AUGMENT.items():
            if key == "augment" or key in out:
                continue
            if key in {"close_mosaic", "max_det"}:
                out[key] = int(default)
            elif key == "multi_scale":
                out[key] = bool(default)
            else:
                out[key] = float(default)
    return out


def run_train(
    *,
    data_yaml: Path,
    model_path: str,
    project: Path,
    device: str,
    imgsz: int,
    epochs: int,
    batch: int,
    patience: int,
    workers: int,
    task: str = "detect",
    single_cls: bool = False,
    run_name: str | None = None,
    extra_kwargs: dict[str, Any] | None = None,
) -> None:
    from ultralytics import YOLO

    use_amp = device != "mps"
    kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "device": device,
        "workers": workers,
        "patience": patience,
        "project": str(project),
        "amp": use_amp,
        "single_cls": single_cls,
    }
    if run_name:
        kwargs["name"] = run_name
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    logger.info(
        "开始训练 task=%s yaml=%s model=%s device=%s imgsz=%s epochs=%s batch=%s single_cls=%s "
        "aug(augment=%s mosaic=%s mixup=%s scale=%s erasing=%s)",
        task,
        data_yaml,
        model_path,
        device,
        imgsz,
        epochs,
        batch,
        single_cls,
        kwargs.get("augment"),
        kwargs.get("mosaic"),
        kwargs.get("mixup"),
        kwargs.get("scale"),
        kwargs.get("erasing"),
    )
    model = YOLO(model_path, task=task)
    model.train(**kwargs)


def main(
    *,
    source_data_root: Path,
    output_dir: Path,
    train_classes: list[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
    keep_empty_labels: bool,
    copy_images: bool,
    skip_prepare: bool,
    prepare_only: bool,
    model_path: str,
    imgsz: int,
    epochs: int,
    batch: int,
    patience: int,
    workers: int,
    device: str,
    single_cls: bool = False,
    label_format: str = "detect",
    task: str = "detect",
    split_each_dir: bool = True,
    voc_paths: list[Path] | None = None,
    run_name: str | None = None,
    extra_kwargs: dict[str, Any] | None = None,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not skip_prepare:
        stats = prepare_dataset(
            source_root=source_data_root,
            output_dir=output_dir,
            train_classes=train_classes,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            keep_empty_labels=keep_empty_labels,
            copy_images=copy_images,
            label_format=label_format,  # type: ignore[arg-type]
            voc_paths=voc_paths,
            split_each_dir=split_each_dir,
        )
        logger.info("数据集统计:\n%s", json.dumps(stats, ensure_ascii=False, indent=2))
        if prepare_only:
            logger.info("prepare_only=true，跳过训练")
            return
    elif prepare_only:
        raise ValueError("prepare_only=true 时须设 skip_prepare=false 以执行数据准备")
    else:
        stats_path = output_dir / "stats.json"
        if stats_path.is_file():
            logger.info(
                "使用已预处理数据集 dir=%s\n%s",
                output_dir,
                stats_path.read_text(encoding="utf-8"),
            )
        else:
            logger.info("使用已预处理数据集 dir=%s", output_dir)

    data_yaml = sync_data_yaml_path(output_dir)
    resolved_device = device.strip() or resolve_device()
    run_train(
        data_yaml=data_yaml,
        model_path=model_path,
        project=output_dir,
        device=resolved_device,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        patience=patience,
        workers=workers,
        task=task,
        single_cls=single_cls,
        run_name=run_name,
        extra_kwargs=extra_kwargs,
    )


def main_from_config(config_path: str | Path | None = None) -> None:
    cfg_path = Path(config_path).expanduser().resolve() if config_path else DEFAULT_TRAIN_CONFIG
    cfg, data_cfg_dir, _cfg_dir = load_merged_cfg(cfg_path)
    logger.info("公共配置目录: %s", data_cfg_dir)
    logger.info("入口配置: %s", cfg_path)

    train_cfg = dict(cfg.get("train") or {})
    val_ratio, test_ratio, seed = split_ratios(cfg)
    source_root, voc_paths = resolve_prepare_source(cfg)
    output_raw = cfg.get("output_dir")
    if not output_raw:
        raise ValueError("缺少 output_dir（应在 data_cfg.json 或入口 train_config 中配置）")
    run_name = train_cfg.get("name")
    aug_kwargs = build_augment_train_kwargs(train_cfg)
    opt_kwargs: dict[str, Any] = dict(aug_kwargs)
    for key in (
        "optimizer",
        "lr0",
        "lrf",
        "cos_lr",
        "momentum",
        "weight_decay",
        "warmup_epochs",
    ):
        if key in train_cfg:
            opt_kwargs[key] = train_cfg[key]
    main(
        source_data_root=source_root,
        output_dir=Path(str(output_raw)),
        train_classes=list(cfg.get("train_classes") or []),
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        split_each_dir=split_each_dir_enabled(cfg),
        voc_paths=voc_paths,
        keep_empty_labels=bool(cfg.get("keep_empty_labels", False)),
        copy_images=bool(cfg.get("copy_images", True)),
        skip_prepare=bool(cfg.get("skip_prepare", False)),
        prepare_only=bool(cfg.get("prepare_only", False)),
        model_path=str(cfg.get("model_path") or "yolo11l.pt"),
        imgsz=int(train_cfg.get("imgsz", 1024)),
        epochs=int(train_cfg.get("epochs", 200)),
        batch=int(train_cfg.get("batch", 4)),
        patience=int(train_cfg.get("patience", 25)),
        workers=int(train_cfg.get("workers", 2)),
        device=str(cfg.get("device") or ""),
        single_cls=bool(cfg.get("single_cls", False)),
        run_name=str(run_name) if run_name else None,
        extra_kwargs=opt_kwargs or None,
    )


if __name__ == "__main__":
    # 客户机 nohup .../python3 train.py > d.log 2>&1 &
    # 改数据/output_dir：train_detect_cfg/data_cfg.json；改超参：train_detect_yolo/train_config.json
    CONFIG_PATH = str(DEFAULT_TRAIN_CONFIG)
    main_from_config(CONFIG_PATH)
