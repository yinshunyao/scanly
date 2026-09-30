#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""封边缺陷检测 core：对已预处理数据集的测试集（默认）做 COCO mAP（.pth / .onnx）。"""
from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

_TRAIN_DIR = Path(__file__).resolve().parent
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

from test import assert_split_ready, resolve_model_path  # noqa: E402
from train import resolve_device, sync_data_yaml_path  # noqa: E402
from train_core import (  # noqa: E402
    DETECT_CORE_ROOT,
    read_class_names,
    solver_device,
    yolo_split_to_coco,
)

logger = logging.getLogger(__name__)


def _eval_resize_ops(imgsz: int) -> list[dict]:
    return [
        {"type": "Resize", "size": [imgsz, imgsz]},
        {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
    ]


def ensure_split_coco(
    staging: Path,
    class_names: list[str],
    split: str,
    *,
    single_cls: bool,
) -> tuple[Path, Path]:
    """写出指定划分的 COCO json，返回 (img_folder, ann_file)。"""
    allowed = {"train", "val", "test"}
    if split not in allowed:
        raise ValueError(f"SPLIT 须为 {sorted(allowed)} 之一，当前: {split}")
    payload = yolo_split_to_coco(staging, split, class_names, single_cls=single_cls)
    n_img = len(payload["images"])
    if n_img == 0:
        raise FileNotFoundError(
            f"划分 {split} 无图像，无法写 COCO: {staging / 'images' / split}"
        )
    ann_dir = staging / "coco" / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    ann_file = ann_dir / f"instances_{split}.json"
    ann_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "COCO %s: images=%d boxes=%d -> %s",
        split,
        n_img,
        len(payload["annotations"]),
        ann_file,
    )
    return staging / "images" / split, ann_file


def build_eval_yaml_update(
    *,
    names: list[str],
    imgsz: int,
    batch: int,
    workers: int,
    device: str,
    seed: int,
    run_dir: Path,
    img_folder: Path,
    ann_file: Path,
    resume: str | None = None,
) -> dict[str, Any]:
    """构造只评估用的 YAMLConfig 覆盖项；val_dataloader 指向给定划分。"""
    val_ds = {
        "img_folder": str(img_folder),
        "ann_file": str(ann_file),
        "transforms": {"ops": _eval_resize_ops(imgsz)},
    }
    update: dict[str, Any] = {
        "num_classes": len(names),
        "remap_mscoco_category": False,
        "eval_spatial_size": [imgsz, imgsz],
        "output_dir": str(run_dir),
        "use_amp": device == "cuda",
        "device": device,
        "seed": seed,
        "print_freq": 50,
        "sync_bn": False,
        "PResNet": {"pretrained": False},
        "val_dataloader": {
            "total_batch_size": batch,
            "num_workers": workers,
            "dataset": val_ds,
        },
        "tuning": None,
    }
    if resume:
        update["resume"] = str(Path(resume).expanduser().resolve())
    return update

_COCO_STAT_NAMES = (
    "AP",
    "AP50",
    "AP75",
    "APs",
    "APm",
    "APl",
    "AR1",
    "AR10",
    "AR100",
    "ARs",
    "ARm",
    "ARl",
)


def detect_backend(model_path: Path) -> str:
    suffix = model_path.suffix.lower()
    if suffix == ".onnx":
        return "onnx"
    if suffix == ".pth":
        return "pth"
    if suffix == ".pt":
        raise ValueError(
            f"MODEL_PATH 为 YOLO 风格 .pt: {model_path}；"
            "请用 scanly/train/test.py 评估 YOLO，"
            "或改为 RT-DETRv2 的 .pth / .onnx"
        )
    raise ValueError(f"不支持的权重后缀 {model_path.suffix}，须为 .pth 或 .onnx: {model_path}")


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


def _mean_valid(arr: Any) -> float | None:
    import numpy as np

    data = np.asarray(arr, dtype=np.float64)
    valid = data[np.isfinite(data) & (data > -1.0)]
    if valid.size == 0:
        return None
    return float(valid.mean())


def _as_seq(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return []


def _float_stats(coco_eval: Any) -> list[float]:
    out: list[float] = []
    for item in _as_seq(getattr(coco_eval, "stats", None)):
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            continue
    return out


def _coco_gt(coco_eval: Any) -> Any:
    return getattr(coco_eval, "cocoGt", None) or getattr(coco_eval, "coco_gt", None)


def per_class_maps(coco_eval: Any, class_names: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = getattr(coco_eval, "eval", None) or {}
    precision = payload.get("precision") if isinstance(payload, dict) else None
    if precision is None:
        return {}, {}
    params = coco_eval.params
    cat_ids = [int(x) for x in _as_seq(getattr(params, "catIds", None))]
    iou_thrs = [float(x) for x in _as_seq(getattr(params, "iouThrs", None))]
    try:
        t50 = next(i for i, thr in enumerate(iou_thrs) if abs(thr - 0.5) < 1e-6)
    except StopIteration:
        t50 = 0
    id_to_name: dict[int, str] = {}
    gt = _coco_gt(coco_eval)
    if gt is not None and hasattr(gt, "loadCats") and cat_ids:
        for cat in gt.loadCats(cat_ids):
            id_to_name[int(cat["id"])] = str(cat["name"])
    map50_95: dict[str, Any] = {}
    map50: dict[str, Any] = {}
    for idx, cat_id in enumerate(cat_ids):
        name = id_to_name.get(cat_id)
        if not name and idx < len(class_names):
            name = class_names[idx]
        if not name:
            name = str(cat_id)
        map50_95[name] = _mean_valid(precision[:, :, idx, 0, -1])
        map50[name] = _mean_valid(precision[t50, :, idx, 0, -1])
    return map50, map50_95


def metrics_from_coco_eval(
    coco_eval: Any,
    *,
    class_names: list[str],
    backend: str,
    split: str,
    model_path: Path,
    save_dir: Path,
) -> dict[str, Any]:
    stats = _float_stats(coco_eval)
    results = {
        _COCO_STAT_NAMES[i]: stats[i] if i < len(stats) else None
        for i in range(len(_COCO_STAT_NAMES))
    }
    per50, per5095 = per_class_maps(coco_eval, class_names)
    return {
        "backend": backend,
        "split": split,
        "model": str(model_path),
        "map50": results.get("AP50"),
        "map50_95": results.get("AP"),
        "map75": results.get("AP75"),
        "ar100": results.get("AR100"),
        "results": _jsonable(results),
        "per_class_map50": _jsonable(per50),
        "per_class_map50_95": _jsonable(per5095),
        "save_dir": str(save_dir),
    }


def write_metrics(
    summary: dict[str, Any],
    save_dir: Path,
    filename: str = "test_metrics.json",
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / filename
    metrics_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics_path


def _make_yaml_config(
    *,
    model_yml: Path,
    names: list[str],
    imgsz: int,
    batch: int,
    workers: int,
    device: str,
    seed: int,
    run_dir: Path,
    img_folder: Path,
    ann_file: Path,
    resume: str | None,
):
    from src.core import YAMLConfig

    update = build_eval_yaml_update(
        names=names,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=device,
        seed=seed,
        run_dir=run_dir,
        img_folder=img_folder,
        ann_file=ann_file,
        resume=resume,
    )
    return YAMLConfig(str(model_yml), **update)


def run_val_pth(ycfg) -> Any:
    from src.misc import dist_utils
    from src.solver import TASKS

    _ = ycfg.model
    solver = TASKS[ycfg.yaml_cfg["task"]](ycfg)
    solver.val()
    if not dist_utils.is_main_process():
        return None
    return solver.evaluator.coco_eval["bbox"]


def _ort_session(model_path: Path, device: str):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "评估 ONNX 需要 onnxruntime；请安装 "
            "scanly/train/detect_core/requirements.txt"
        ) from exc

    available = list(ort.get_available_providers())
    providers = ["CPUExecutionProvider"]
    if device == "cuda" and "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif device == "cuda":
        logger.warning(
            "ONNX 评估无 CUDAExecutionProvider，回退 CPU；available=%s",
            available,
        )
    elif device == "mps":
        logger.warning("ONNX 评估不使用 mps，回退 CPU")
    logger.info("ONNX Runtime providers=%s", providers)
    return ort.InferenceSession(str(model_path), providers=providers)


def run_val_onnx(ycfg, model_path: Path, device: str) -> Any:
    import numpy as np
    import torch

    sess = _ort_session(model_path, device)
    input_names = [item.name for item in sess.get_inputs()]
    loader = ycfg.val_dataloader
    evaluator = ycfg.evaluator
    evaluator.cleanup()

    for step, (samples, targets) in enumerate(loader, start=1):
        orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        images = np.ascontiguousarray(
            samples.detach().cpu().numpy(), dtype=np.float32
        )
        sizes = np.ascontiguousarray(
            orig_sizes.detach().cpu().numpy(), dtype=np.int64
        )
        feed: dict[str, Any] = {}
        if "images" in input_names:
            feed["images"] = images
        else:
            feed[input_names[0]] = images
        if "orig_target_sizes" in input_names:
            feed["orig_target_sizes"] = sizes
        elif len(input_names) > 1:
            feed[input_names[1]] = sizes
        labels, boxes, scores = sess.run(None, feed)
        labels_t = torch.as_tensor(np.asarray(labels))
        boxes_t = torch.as_tensor(np.asarray(boxes))
        scores_t = torch.as_tensor(np.asarray(scores))
        results = [
            dict(labels=labels_t[i], boxes=boxes_t[i], scores=scores_t[i])
            for i in range(len(targets))
        ]
        res = {target["image_id"].item(): output for target, output in zip(targets, results)}
        evaluator.update(res)
        if step == 1 or step % 10 == 0:
            logger.info("onnx eval batch=%d n=%d", step, len(targets))

    evaluator.synchronize_between_processes()
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.accumulate()
        evaluator.summarize()
    return evaluator.coco_eval["bbox"]


def run_val(
    *,
    output_dir: Path,
    model_path: Path,
    model_yml: Path,
    project: Path,
    run_name: str,
    device: str,
    imgsz: int,
    batch: int,
    workers: int,
    split: str,
    seed: int,
    single_cls: bool,
    metrics_filename: str = "test_metrics.json",
) -> dict[str, Any]:
    backend = detect_backend(model_path)
    if not model_yml.is_file():
        raise FileNotFoundError(f"未找到 MODEL_YML: {model_yml}")

    class_names = read_class_names(output_dir)
    names = ["defect"] if single_cls else list(class_names)
    img_folder, ann_file = ensure_split_coco(
        output_dir, class_names, split, single_cls=single_cls
    )
    save_dir = project / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    from src.misc import dist_utils

    dist_utils.setup_distributed(print_rank=0, print_method="builtin", seed=seed)
    try:
        ycfg = _make_yaml_config(
            model_yml=model_yml,
            names=names,
            imgsz=imgsz,
            batch=batch,
            workers=workers,
            device=device,
            seed=seed,
            run_dir=save_dir,
            img_folder=img_folder,
            ann_file=ann_file,
            resume=str(model_path) if backend == "pth" else None,
        )
        logger.info(
            "开始验证 backend=%s split=%s yaml=%s model=%s device=%s imgsz=%s batch=%s single_cls=%s",
            backend,
            split,
            model_yml,
            model_path,
            device,
            imgsz,
            batch,
            single_cls,
        )
        if backend == "pth":
            coco_eval = run_val_pth(ycfg)
        else:
            coco_eval = run_val_onnx(ycfg, model_path, device)
    finally:
        dist_utils.cleanup()

    if coco_eval is None:
        raise RuntimeError("非主进程未得到 COCO 评估结果")
    summary = metrics_from_coco_eval(
        coco_eval,
        class_names=names,
        backend=backend,
        split=split,
        model_path=model_path,
        save_dir=save_dir,
    )
    metrics_path = write_metrics(summary, save_dir, filename=metrics_filename)
    logger.info(
        "验证完成 split=%s backend=%s mAP50=%s mAP50-95=%s metrics=%s",
        split,
        backend,
        summary.get("map50"),
        summary.get("map50_95"),
        metrics_path,
    )
    return summary


def main(
    *,
    output_dir: Path,
    model_path: Path,
    model_yml: Path,
    train_project: Path,
    run_name: str,
    imgsz: int,
    batch: int,
    workers: int,
    device: str,
    split: str,
    seed: int,
    single_cls: bool = False,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    data_yaml = sync_data_yaml_path(output_dir)
    assert_split_ready(output_dir, data_yaml, split)
    resolved_device = solver_device(device.strip() or resolve_device())

    prev_cwd = Path.cwd()
    if str(DETECT_CORE_ROOT) not in sys.path:
        sys.path.insert(0, str(DETECT_CORE_ROOT))
    os.chdir(DETECT_CORE_ROOT)
    try:
        run_val(
            output_dir=output_dir,
            model_path=model_path,
            model_yml=model_yml,
            project=train_project,
            run_name=run_name,
            device=resolved_device,
            imgsz=imgsz,
            batch=batch,
            workers=workers,
            split=split,
            seed=seed,
            single_cls=single_cls,
        )
    finally:
        os.chdir(prev_cwd)


if __name__ == "__main__":
    # /home/beyond/.conda/envs/yolo11/bin/python3 test_core.py > test_core.log 2>&1 &
    TRAIN_DIR = Path(__file__).resolve().parent

    # —— 数据 / 训练对齐 train_core.py ——
    OUTPUT_DIR = TRAIN_DIR / "output" / "cam2-0911-1024-train"
    IMGSZ = 1024
    BATCH = 10
    WORKERS = 8
    DEVICE = ""  # 空则自动选择 cuda / mps / cpu
    SINGLE_CLS = False  # 须与对应训练权重一致
    SPLIT = "test"  # 可改为 val
    SEED = 42

    # —— 权重与输出 ——
    MODEL_PATH = TRAIN_DIR / "runs" / "detect_core" / "best.pth"  # 或 best.onnx
    MODEL_YML = DETECT_CORE_ROOT / "configs" / "core" / "core_r50vd_scanly.yml"
    TRAIN_PROJECT = TRAIN_DIR / "runs"
    RUN_NAME = "detect_core-test"

    main(
        output_dir=OUTPUT_DIR,
        model_path=resolve_model_path(MODEL_PATH, train_dir=TRAIN_DIR),
        model_yml=MODEL_YML,
        train_project=TRAIN_PROJECT,
        run_name=RUN_NAME,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        split=SPLIT,
        seed=SEED,
        single_cls=SINGLE_CLS,
    )
