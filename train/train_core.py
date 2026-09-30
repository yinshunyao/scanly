#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""封边缺陷检测 core 训练：数据准备与 train.py 对齐，再转 COCO 后训练。"""
from __future__ import annotations

import inspect
import json
import logging
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

_TRAIN_DIR = Path(__file__).resolve().parent
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

from prepare_dataset import prepare_dataset  # noqa: E402
from train import assert_prepared_dataset, resolve_device, sync_data_yaml_path  # noqa: E402
from train_detect_cfg.load_cfg import (  # noqa: E402
    load_merged_cfg,
    resolve_source_root,
    split_ratios,
)

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")
DETECT_CORE_ROOT = _TRAIN_DIR / "detect_core"
DEFAULT_TRAIN_CONFIG = _TRAIN_DIR / "train_detect_core" / "train_config.json"


def _line_buffer_stdio() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass


def call_prepare_dataset(**kwargs: Any) -> dict[str, Any]:
    """按当前 prepare_dataset 签名传参，兼容尚未升级的副本（可能无 test_ratio）。"""
    params = inspect.signature(prepare_dataset).parameters
    accepted = {key: value for key, value in kwargs.items() if key in params}
    skipped = [key for key in kwargs if key not in params]
    if skipped:
        logger.warning(
            "当前 prepare_dataset 不支持 %s，已忽略；建议同步最新 prepare_dataset.py",
            ", ".join(skipped),
        )
    return prepare_dataset(**accepted)


def solver_device(raw: str) -> str:
    device = (raw or "").strip() or resolve_device()
    if device.startswith("cuda"):
        return "cuda"
    return device


def read_class_names(output_dir: Path) -> list[str]:
    classes_txt = output_dir / "classes.txt"
    if classes_txt.is_file():
        names = [
            line.strip()
            for line in classes_txt.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if names:
            return names
    data_yaml = assert_prepared_dataset(output_dir)
    names: list[str] = []
    in_names = False
    for raw in data_yaml.read_text(encoding="utf-8").splitlines():
        if raw.startswith("names:"):
            in_names = True
            rest = raw[6:].strip()
            if rest.startswith("["):
                inner = rest.strip("[]")
                names = [p.strip().strip("'\"") for p in inner.split(",") if p.strip()]
                break
            continue
        if not in_names:
            continue
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if raw.startswith("  - "):
            names.append(raw[4:].strip().strip("'\""))
            continue
        if raw.startswith("  ") and ":" in stripped:
            names.append(stripped.split(":", 1)[1].strip().strip("'\""))
            continue
        if not raw.startswith(" "):
            break
    if not names:
        raise ValueError(f"无法从 {output_dir} 读取类别名（classes.txt / data.yaml names）")
    return names


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size


def _iter_split_images(img_dir: Path) -> list[Path]:
    if not img_dir.is_dir():
        return []
    return sorted(
        p
        for p in img_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("._")
    )


def yolo_split_to_coco(
    staging: Path,
    split: str,
    class_names: list[str],
    *,
    single_cls: bool,
) -> dict[str, Any]:
    img_dir = staging / "images" / split
    lbl_dir = staging / "labels" / split
    images: list[dict] = []
    annotations: list[dict] = []
    ann_id = 1
    n_cls = 1 if single_cls else len(class_names)
    for img_id, img_path in enumerate(_iter_split_images(img_dir), start=1):
        w, h = _image_size(img_path)
        images.append(
            {
                "id": img_id,
                "file_name": img_path.name,
                "width": int(w),
                "height": int(h),
            }
        )
        txt = lbl_dir / f"{img_path.stem}.txt"
        if not txt.is_file():
            continue
        for raw in txt.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(float(parts[0]))
                xc, yc, bw, bh = (
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                )
            except ValueError:
                continue
            if single_cls:
                cls_id = 0
            elif cls_id < 0 or cls_id >= n_cls:
                continue
            box_w = bw * w
            box_h = bh * h
            x = (xc - bw / 2.0) * w
            y = (yc - bh / 2.0) * h
            x = max(0.0, min(float(w), x))
            y = max(0.0, min(float(h), y))
            box_w = max(0.0, min(float(w) - x, box_w))
            box_h = max(0.0, min(float(h) - y, box_h))
            if box_w <= 1.0 or box_h <= 1.0:
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls_id,
                    "bbox": [x, y, box_w, box_h],
                    "area": box_w * box_h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    if single_cls:
        categories = [{"id": 0, "name": "defect", "supercategory": "defect"}]
    else:
        categories = [
            {"id": i, "name": name, "supercategory": "defect"}
            for i, name in enumerate(class_names)
        ]
    return {
        "info": {"description": "scanly detect_core from yolo staging"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def write_coco_from_yolo(
    staging: Path,
    coco_root: Path,
    class_names: list[str],
    *,
    single_cls: bool,
) -> tuple[Path, Path]:
    ann_dir = coco_root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for split in SPLITS:
        payload = yolo_split_to_coco(staging, split, class_names, single_cls=single_cls)
        n_img = len(payload["images"])
        n_box = len(payload["annotations"])
        logger.info("COCO %s: images=%d boxes=%d", split, n_img, n_box)
        if n_img == 0 and split != "test":
            continue
        out = ann_dir / f"instances_{split}.json"
        out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        written[split] = out
    if "train" not in written:
        raise RuntimeError(f"staging 无 train 图片: {staging / 'images' / 'train'}")
    if "val" not in written:
        logger.warning("val 为空，复用 train 的 COCO json")
        written["val"] = written["train"]
    return written["train"], written["val"]


def allocate_run_dir(base: Path, *, resume: str | None) -> Path:
    if resume:
        rp = Path(resume).expanduser()
        if rp.is_dir():
            return rp.resolve()
        return rp.parent.resolve()
    if not base.exists():
        return base
    parent = base.parent
    stem = base.name
    n = 2
    while True:
        cand = parent / f"{stem}_{n}"
        if not cand.exists():
            return cand
        n += 1


def ensure_pretrained(path: Path, url: str) -> Path:
    if path.is_file():
        return path
    if not url:
        raise FileNotFoundError(f"预训练不存在且未配置 TUNING_URL: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("下载预训练 %s -> %s", url, path)
    urllib.request.urlretrieve(url, str(path))
    return path


def _resize_ops(imgsz: int, *, train: bool) -> list[dict]:
    if not train:
        return [
            {"type": "Resize", "size": [imgsz, imgsz]},
            {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
        ]
    return [
        {"type": "RandomPhotometricDistort", "p": 0.5},
        {"type": "RandomZoomOut", "fill": 0},
        {"type": "RandomIoUCrop", "p": 0.8},
        {"type": "SanitizeBoundingBoxes", "min_size": 1},
        {"type": "RandomHorizontalFlip"},
        {"type": "Resize", "size": [imgsz, imgsz]},
        {"type": "SanitizeBoundingBoxes", "min_size": 1},
        {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
        {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True},
    ]


def _release_train_runtime(cfg) -> None:
    for attr in (
        "_ema",
        "_optimizer",
        "_lr_scheduler",
        "_lr_warmup_scheduler",
        "_train_dataloader",
        "_val_dataloader",
        "_criterion",
        "_scaler",
        "_evaluator",
        "_writer",
        "_model",
        "_postprocessor",
    ):
        if hasattr(cfg, attr):
            setattr(cfg, attr, None)
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def resolve_export_checkpoint(output_dir: Path) -> Path | None:
    best = output_dir / "best.pth"
    if best.is_file():
        return best
    last = output_dir / "last.pth"
    if last.is_file():
        return last
    return None


def run_post_train_eval(
    *,
    output_dir: Path,
    run_dir: Path,
    model_yml: Path,
    checkpoint: Path,
    imgsz: int,
    batch: int,
    workers: int,
    device: str,
    seed: int,
    single_cls: bool,
    test_ratio: float,
    val_split: str,
) -> None:
    """fit 结束后先 val、再按 TEST_RATIO 与 test 图像决定是否 test。延迟 import 避免循环依赖。"""
    sys.path.insert(0, str(_TRAIN_DIR))
    from test_core import run_val

    logger.info("训练后评估：val split=%s ckpt=%s", val_split, checkpoint)
    run_val(
        output_dir=output_dir,
        model_path=checkpoint,
        model_yml=model_yml,
        project=run_dir.parent,
        run_name=run_dir.name,
        device=device,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        split=val_split,
        seed=seed,
        single_cls=single_cls,
        metrics_filename="val_metrics.json",
    )
    if test_ratio <= 0:
        logger.info("TEST_RATIO=%s <= 0，跳过 test 评估", test_ratio)
        return
    test_img_dir = output_dir / "images" / "test"
    if not _iter_split_images(test_img_dir):
        logger.warning(
            "TEST_RATIO=%s 但无 test 图像，跳过 test 评估: %s",
            test_ratio,
            test_img_dir,
        )
        return
    logger.info("训练后评估：test ckpt=%s", checkpoint)
    run_val(
        output_dir=output_dir,
        model_path=checkpoint,
        model_yml=model_yml,
        project=run_dir.parent,
        run_name=run_dir.name,
        device=device,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        split="test",
        seed=seed,
        single_cls=single_cls,
        metrics_filename="test_metrics.json",
    )


def run_train(
    *,
    output_dir: Path,
    class_names: list[str],
    train_project: Path,
    run_prefix: str,
    model_yml: Path,
    tuning_path: Path | None,
    tuning_url: str,
    imgsz: int,
    epochs: int,
    batch: int,
    patience: int,
    workers: int,
    device: str,
    seed: int,
    close_mosaic: int,
    val_fitness_metric: str,
    single_cls: bool,
    test_only: bool,
    resume: str | None,
    export_onnx: bool,
    lr: float,
    backbone_lr: float,
    weight_decay: float,
    test_ratio: float,
) -> None:
    sys.path.insert(0, str(DETECT_CORE_ROOT))
    os.chdir(DETECT_CORE_ROOT)

    from src.core import YAMLConfig
    from src.misc import dist_utils
    from src.solver import TASKS
    from src.solver.det_solver import normalize_val_fitness_metric

    names = ["defect"] if single_cls else list(class_names)
    coco_root = output_dir / "coco"
    train_json, val_json = write_coco_from_yolo(
        output_dir, coco_root, class_names, single_cls=single_cls
    )
    train_img = str(output_dir / "images" / "train")
    val_split = (
        "val"
        if (output_dir / "images" / "val").is_dir()
        and _iter_split_images(output_dir / "images" / "val")
        else "train"
    )
    val_img = str(output_dir / "images" / val_split)

    run_dir = allocate_run_dir(train_project / run_prefix, resume=resume)
    run_dir.mkdir(parents=True, exist_ok=True)
    stop_epoch = max(0, epochs - max(0, close_mosaic))
    use_amp = device == "cuda"
    metric = normalize_val_fitness_metric(val_fitness_metric)
    dist_utils.setup_distributed(print_rank=0, print_method="builtin", seed=seed)

    tuning = None
    if tuning_path and not resume:
        tuning = str(ensure_pretrained(tuning_path, tuning_url))

    update: dict[str, Any] = {
        "num_classes": len(names),
        "remap_mscoco_category": False,
        "eval_spatial_size": [imgsz, imgsz],
        "epoches": epochs,
        "output_dir": str(run_dir),
        "use_amp": use_amp,
        "device": device,
        "seed": seed,
        "print_freq": 50,
        "checkpoint_freq": 5,
        "val_fitness_metric": metric,
        "patience": max(0, patience),
        "sync_bn": False,
        "train_dataloader": {
            "total_batch_size": batch,
            "num_workers": workers,
            "dataset": {
                "img_folder": train_img,
                "ann_file": str(train_json),
                "transforms": {
                    "ops": _resize_ops(imgsz, train=True),
                    "policy": {"epoch": stop_epoch},
                },
            },
            "collate_fn": {"scales": None, "stop_epoch": stop_epoch},
        },
        "val_dataloader": {
            "total_batch_size": batch,
            "num_workers": workers,
            "dataset": {
                "img_folder": val_img,
                "ann_file": str(val_json),
                "transforms": {"ops": _resize_ops(imgsz, train=False)},
            },
        },
        "optimizer": {
            "lr": lr,
            "weight_decay": weight_decay,
            "params": [
                {
                    "params": r"^(?=.*backbone)(?!.*norm).*$",
                    "lr": backbone_lr,
                },
                {
                    "params": r"^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$",
                    "weight_decay": 0.0,
                },
            ],
        },
    }
    if tuning:
        update["tuning"] = tuning
        update["PResNet"] = {"pretrained": False}
    if resume:
        update["resume"] = str(Path(resume).expanduser().resolve())
        update["tuning"] = None

    logger.info(
        "detect_core: nc=%d imgsz=%d epochs=%d batch=%d device=%s amp=%s yml=%s tuning=%s output_dir=%s val_fitness_metric=%s patience=%d",
        len(names),
        imgsz,
        epochs,
        batch,
        device,
        use_amp,
        model_yml,
        tuning,
        run_dir,
        metric,
        patience,
    )
    ycfg = YAMLConfig(str(model_yml), **update)
    _ = ycfg.model
    solver = TASKS[ycfg.yaml_cfg["task"]](ycfg)
    if test_only:
        logger.info("开始 val：加载 resume 权重并评估")
        solver.val()
        dist_utils.cleanup()
        return

    logger.info("开始 fit")
    solver.fit()
    del solver
    _release_train_runtime(ycfg)
    dist_utils.cleanup()
    ckpt = resolve_export_checkpoint(run_dir)
    if ckpt is None:
        logger.warning("训练结束但无 best.pth/last.pth，跳过训练后评估与 ONNX: %s", run_dir)
        return
    run_post_train_eval(
        output_dir=output_dir,
        run_dir=run_dir,
        model_yml=model_yml,
        checkpoint=ckpt,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=device,
        seed=seed,
        single_cls=single_cls,
        test_ratio=test_ratio,
        val_split=val_split,
    )
    if export_onnx:
        from src.misc.onnx_export import export_to_onnx

        out = ckpt.with_suffix(".onnx")
        logger.info("导出 ONNX: %s -> %s (imgsz=%d)", ckpt, out, imgsz)
        export_to_onnx(ycfg, out, imgsz, checkpoint=ckpt, check=True, simplify=False)
        logger.info("ONNX 已写出: %s", out)


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
    train_project: Path,
    run_prefix: str,
    model_yml: Path,
    tuning_path: Path | None,
    tuning_url: str,
    imgsz: int,
    epochs: int,
    batch: int,
    patience: int,
    workers: int,
    device: str,
    single_cls: bool,
    close_mosaic: int,
    val_fitness_metric: str,
    test_only: bool,
    resume: str | None,
    export_onnx: bool,
    lr: float,
    backbone_lr: float,
    weight_decay: float,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _line_buffer_stdio()

    if test_only and not resume:
        raise ValueError("TEST_ONLY=True 时须设置 RESUME 为 checkpoint 路径")
    if prepare_only and skip_prepare:
        raise ValueError("PREPARE_ONLY=True 时须设 SKIP_PREPARE=False 以执行数据准备")

    if not skip_prepare:
        stats = call_prepare_dataset(
            source_root=source_data_root,
            output_dir=output_dir,
            train_classes=train_classes,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            keep_empty_labels=keep_empty_labels,
            copy_images=copy_images,
            label_format="detect",
        )
        logger.info("数据集统计:\n%s", json.dumps(stats, ensure_ascii=False, indent=2))
        if prepare_only:
            logger.info("PREPARE_ONLY=True，跳过训练")
            return
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

    sync_data_yaml_path(output_dir)
    class_names = read_class_names(output_dir)
    resolved_device = solver_device(device)
    run_train(
        output_dir=output_dir,
        class_names=class_names,
        train_project=train_project,
        run_prefix=run_prefix,
        model_yml=model_yml,
        tuning_path=tuning_path,
        tuning_url=tuning_url,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        patience=patience,
        workers=workers,
        device=resolved_device,
        seed=seed,
        close_mosaic=close_mosaic,
        val_fitness_metric=val_fitness_metric,
        single_cls=single_cls,
        test_only=test_only,
        resume=resume,
        export_onnx=export_onnx,
        lr=lr,
        backbone_lr=backbone_lr,
        weight_decay=weight_decay,
        test_ratio=test_ratio,
    )


def main_from_config(
    config_path: str | Path | None = None,
    *,
    test_only: bool = False,
    resume: str | None = None,
) -> None:
    cfg_path = Path(config_path).expanduser().resolve() if config_path else DEFAULT_TRAIN_CONFIG
    cfg, data_cfg_dir, _cfg_dir = load_merged_cfg(cfg_path)
    logger.info("公共配置目录: %s", data_cfg_dir)
    logger.info("入口配置: %s", cfg_path)

    train_cfg = dict(cfg.get("train") or {})
    val_ratio, test_ratio, seed = split_ratios(cfg)
    tuning_raw = cfg.get("tuning")
    tuning_path = Path(str(tuning_raw)) if tuning_raw else None
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
        train_project=Path(str(cfg.get("train_project") or (_TRAIN_DIR / "runs"))),
        run_prefix=str(cfg.get("run_prefix") or "detect_core"),
        model_yml=Path(str(cfg["model_yml"])),
        tuning_path=tuning_path,
        tuning_url=str(cfg.get("tuning_url") or ""),
        imgsz=int(train_cfg.get("imgsz", 1024)),
        epochs=int(train_cfg.get("epochs", 100)),
        batch=int(train_cfg.get("batch", 4)),
        patience=int(train_cfg.get("patience", 15)),
        workers=int(train_cfg.get("workers", 8)),
        device=str(cfg.get("device") or ""),
        single_cls=bool(cfg.get("single_cls", False)),
        close_mosaic=int(cfg.get("close_mosaic", 10)),
        val_fitness_metric=str(cfg.get("val_fitness_metric") or "ap50"),
        test_only=test_only,
        resume=resume,
        export_onnx=bool(cfg.get("export_onnx", True)),
        lr=float(train_cfg.get("lr", 0.0001)),
        backbone_lr=float(train_cfg.get("backbone_lr", 0.00001)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0001)),
    )


if __name__ == "__main__":
    # 改数据：train_detect_cfg/data_cfg.json；改超参/骨干：train_detect_core/train_config.json
    import multiprocessing

    multiprocessing.freeze_support()
    CONFIG_PATH = str(DEFAULT_TRAIN_CONFIG)
    TEST_ONLY = False
    RESUME = None  # 例如 "runs/core_r18/last.pth"
    main_from_config(CONFIG_PATH, test_only=TEST_ONLY, resume=RESUME)
