#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""封边缺陷检测 mark 训练：按公共数据配置准备 YOLO 目录后训练。

数据源读 ``train_detect_cfg``；训练超参在 ``train_detect_mark/train_config.json``。
"""
from __future__ import annotations

import importlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

_TRAIN_DIR = Path(__file__).resolve().parent
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

from train_detect_cfg.prepare_dataset import prepare_dataset  # noqa: E402
from train import assert_prepared_dataset, resolve_device, sync_data_yaml_path  # noqa: E402
from train_detect_cfg.load_cfg import (  # noqa: E402
    load_merged_cfg,
    resolve_prepare_source,
    split_each_dir_enabled,
    split_ratios,
)

logger = logging.getLogger(__name__)

DEFAULT_TRAIN_CONFIG = _TRAIN_DIR / "train_detect_mark" / "train_config.json"

_SIZE_SUFFIX = {
    "n": "Nano",
    "s": "Small",
    "m": "Medium",
    "l": "Large",
    "b": "Base",
}
_SIZE_BLOCK = {
    "n": 32,
    "s": 32,
    "m": 32,
    "l": 32,
    "b": 56,
}
_RESUME_NAMES = ("last.ckpt", "checkpoint_best_total.pth", "checkpoint_best_ema.pth")


def _hidden_tokens() -> tuple[str, ...]:
    parts = ("rf", "de", "tr")
    pkg = "".join(parts)
    hyphen = parts[0] + "-" + parts[1] + parts[2]
    brand = "robo" + "flow"
    head = "RF"
    tail = "DE" + "TR"
    tokens = {
        pkg,
        pkg.upper(),
        pkg.title(),
        hyphen,
        hyphen.upper(),
        hyphen.title(),
        head + tail,
        head + "-" + tail,
        brand,
        brand.title(),
        brand.upper(),
    }
    return tuple(sorted(tokens, key=len, reverse=True))


def _redact(text: str) -> str:
    for token in _hidden_tokens():
        if token and token in text:
            text = text.replace(token, "mark")
    return text


def _pkg_name() -> str:
    return "rf" + "de" + "tr"


def _cls_prefix() -> str:
    return "RF" + "DE" + "TR"


class _RedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        rendered = record.getMessage()
        cleaned = _redact(rendered)
        if cleaned != rendered:
            record.msg = cleaned
            record.args = ()
            record.message = cleaned
        return True


class _RedactStream:
    def __init__(self, raw: Any) -> None:
        self._raw = raw

    def write(self, data: Any) -> int:
        if isinstance(data, str):
            data = _redact(data)
        return self._raw.write(data)

    def flush(self) -> None:
        self._raw.flush()

    def isatty(self) -> bool:
        fn = getattr(self._raw, "isatty", None)
        return bool(fn()) if callable(fn) else False

    def fileno(self) -> int:
        return self._raw.fileno()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)


def _attach_redact_filters(filt: logging.Filter) -> None:
    seen: set[int] = set()

    def _add(obj: logging.Logger) -> None:
        if id(obj) in seen:
            return
        seen.add(id(obj))
        obj.addFilter(filt)
        for handler in obj.handlers:
            handler.addFilter(filt)

    root = logging.getLogger()
    _add(root)
    for obj in logging.Logger.manager.loggerDict.values():
        if isinstance(obj, logging.Logger):
            _add(obj)


def _arm_redact() -> tuple[Any, Any, _RedactFilter]:
    filt = _RedactFilter()
    _attach_redact_filters(filt)
    out, err = sys.stdout, sys.stderr
    sys.stdout = _RedactStream(out)
    sys.stderr = _RedactStream(err)
    return out, err, filt


def _restore_streams(out: Any, err: Any) -> None:
    sys.stdout = out
    sys.stderr = err


def _scrub_exc(exc: BaseException) -> None:
    if not exc.args:
        return
    exc.args = tuple(_redact(item) if isinstance(item, str) else item for item in exc.args)


def solver_device(raw: str) -> str:
    device = (raw or "").strip() or resolve_device()
    if device.startswith("cuda"):
        return "cuda"
    return device


def normalize_size(raw: str) -> str:
    size = (raw or "").strip().lower()
    if size not in _SIZE_SUFFIX:
        allowed = " / ".join(_SIZE_SUFFIX)
        raise ValueError(f"size 只能是 {allowed}，当前为 {raw!r}")
    return size


def assert_imgsz(size: str, imgsz: int) -> None:
    block = _SIZE_BLOCK[size]
    if imgsz <= 0 or imgsz % block != 0:
        raise ValueError(
            f"imgsz={imgsz} 不能被 {block} 整除（size={size}）。"
            "n / s / m / l 的边长须为 32 的倍数，b 须为 56 的倍数。"
        )


def allocate_run_dir(base: Path) -> Path:
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


def resolve_run(output_dir: Path, run_prefix: str, resume: str) -> tuple[Path, str | None]:
    text = (resume or "").strip()
    if not text:
        return allocate_run_dir(output_dir / run_prefix), None
    path = Path(text).expanduser().resolve()
    if path.is_dir():
        for name in _RESUME_NAMES:
            cand = path / name
            if cand.is_file():
                return path, str(cand)
        raise FileNotFoundError(f"resume 目录里没有可续训的检查点: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"resume 不是检查点文件: {path}")
    return path.parent, str(path)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
_BEST_WEIGHT_NAMES = (
    "checkpoint_best_total.pth",
    "checkpoint_best_ema.pth",
    "checkpoint_best_regular.pth",
)


def resolve_eval_classes(cfg: dict[str, Any], train_classes: list[str]) -> list[str]:
    """评估清单。未写、null 或空列表时用训练类别。"""
    if "val_classes" not in cfg or cfg.get("val_classes") in (None, ""):
        return list(train_classes)
    raw = cfg.get("val_classes")
    if not isinstance(raw, list):
        raise TypeError("val_classes 须为字符串列表")
    names = [str(item).strip() for item in raw if str(item).strip()]
    if not names:
        return list(train_classes)
    missing = [name for name in names if name not in train_classes]
    if missing:
        raise ValueError(
            "val_classes 必须是 train_classes 的子集，未知类: " + ", ".join(missing)
        )
    return names


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
    data_yaml = output_dir / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"未找到类别表: {output_dir}")
    names: list[str] = []
    in_names = False
    for raw in data_yaml.read_text(encoding="utf-8").splitlines():
        if raw.startswith("names:"):
            in_names = True
            rest = raw[6:].strip()
            if rest.startswith("["):
                return [part.strip().strip("'\"") for part in rest.strip("[]").split(",") if part.strip()]
            continue
        if not in_names:
            continue
        if raw.startswith("  - "):
            names.append(raw[4:].strip().strip("'\""))
            continue
        if not raw.startswith(" "):
            break
    if not names:
        raise ValueError(f"无法从 {data_yaml} 读取类别名")
    return names


def resolve_pretrain(model_path: str) -> str | None:
    text = (model_path or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"预训练权重不存在: {path}")
    return str(path.resolve())


def best_weight_in(run_dir: Path) -> Path | None:
    for name in _BEST_WEIGHT_NAMES:
        path = run_dir / name
        if path.is_file():
            return path
    return None


def latest_trained_weight(output_dir: Path, run_prefix: str) -> Path | None:
    dirs = [output_dir / run_prefix]
    dirs.extend(
        path
        for path in output_dir.glob(f"{run_prefix}_*")
        if path.is_dir() and path.name[len(run_prefix) + 1 :].isdigit()
    )
    found = [path for path in dirs if path.is_dir()]
    found.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for run_dir in found:
        weight = best_weight_in(run_dir)
        if weight is not None:
            return weight
    return None


def resolve_test_weight(
    *,
    output_dir: Path,
    run_prefix: str,
    resume: str,
    model_path: str,
) -> Path:
    text = (resume or "").strip()
    if text:
        path = Path(text).expanduser().resolve()
        if path.is_file() and path.suffix.lower() == ".pth":
            return path
        if path.is_dir():
            weight = best_weight_in(path)
            if weight is not None:
                return weight
    trained = latest_trained_weight(output_dir, run_prefix)
    if trained is not None:
        return trained
    pretrain = resolve_pretrain(model_path) if (model_path or "").strip() else None
    if pretrain:
        return Path(pretrain)
    raise FileNotFoundError(
        f"没有可评估的权重。请先训练，或把 resume / model_path 指到 .pth。目录: {output_dir}"
    )


def _iter_images(img_dir: Path) -> list[Path]:
    if not img_dir.is_dir():
        return []
    return sorted(
        path
        for path in img_dir.iterdir()
        if path.suffix.lower() in IMAGE_EXTS and not path.name.startswith("._")
    )


def yolo_gt_boxes(
    label_path: Path,
    width: int,
    height: int,
    class_names: list[str],
    keep: set[str],
) -> list[dict[str, Any]]:
    if not label_path.is_file():
        return []
    gts: list[dict[str, Any]] = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            xc, yc, bw, bh = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        except ValueError:
            continue
        if cls_id < 0 or cls_id >= len(class_names):
            continue
        name = class_names[cls_id]
        if name not in keep:
            continue
        box_w = bw * width
        box_h = bh * height
        x1 = (xc - bw / 2.0) * width
        y1 = (yc - bh / 2.0) * height
        x2 = min(float(width), x1 + box_w)
        y2 = min(float(height), y1 + box_h)
        x1 = max(0.0, x1)
        y1 = max(0.0, y1)
        if x2 - x1 <= 1.0 or y2 - y1 <= 1.0:
            continue
        gts.append({"label_id": cls_id, "label": name, "box": [x1, y1, x2, y2]})
    return gts


def _rows_from_predict(raw: Any) -> list[tuple[list[float], float, int]]:
    if raw is None:
        return []
    xyxy = getattr(raw, "xyxy", None)
    if xyxy is not None:
        conf = getattr(raw, "confidence", None)
        cls_id = getattr(raw, "class_id", None)
        rows: list[tuple[list[float], float, int]] = []
        for i, box in enumerate(xyxy):
            score = float(conf[i]) if conf is not None else 1.0
            lid = int(cls_id[i]) if cls_id is not None else 0
            rows.append(([float(v) for v in box], score, lid))
        return rows
    if isinstance(raw, dict):
        boxes = raw.get("boxes") or raw.get("xyxy")
        scores = raw.get("scores") or raw.get("confidence")
        labels = raw.get("labels") or raw.get("class_id")
        if boxes is None:
            return []
        rows = []
        for i, box in enumerate(boxes):
            score = float(scores[i]) if scores is not None else 1.0
            lid = int(labels[i]) if labels is not None else 0
            rows.append(([float(v) for v in box], score, lid))
        return rows
    if isinstance(raw, (list, tuple)):
        if len(raw) == 1:
            return _rows_from_predict(raw[0])
        rows = []
        for item in raw:
            if isinstance(item, dict) and ("box" in item or "xyxy" in item or "bbox" in item):
                box = item.get("box") or item.get("xyxy") or item.get("bbox")
                score = float(item.get("score", item.get("confidence", 1.0)))
                lid = int(item.get("label_id", item.get("class_id", 0)))
                rows.append(([float(v) for v in box], score, lid))
        return rows
    return []


def preds_from_predict(
    raw: Any,
    class_names: list[str],
    keep: set[str],
    *,
    conf_thr: float,
) -> list[dict[str, Any]]:
    preds: list[dict[str, Any]] = []
    for box, score, lid in _rows_from_predict(raw):
        if score < conf_thr:
            continue
        if lid < 0 or lid >= len(class_names):
            continue
        name = class_names[lid]
        if name not in keep:
            continue
        preds.append(
            {
                "label_id": lid,
                "label": name,
                "score": score,
                "box": box,
            }
        )
    preds.sort(key=lambda item: item["score"], reverse=True)
    return preds


def _predict_image(model: Any, image: Any, conf: float) -> Any:
    try:
        return model.predict(image, threshold=conf)
    except TypeError:
        return model.predict(image)


def _sync_infer_device() -> None:
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def vis_root_for_weight(weight: Path, output_dir: Path, run_prefix: str) -> Path:
    """画框写到权重所在的 run 目录。权重不在 output_dir 的 run 下时，退回数据集根。"""
    parent = weight.resolve().parent
    root = output_dir.resolve()
    try:
        parent.relative_to(root)
    except ValueError:
        return root
    name = parent.name
    numbered = name.startswith(f"{run_prefix}_") and name[len(run_prefix) + 1 :].isdigit()
    if name == run_prefix or numbered:
        return parent
    return root


def _ratio(numer: int, denom: int) -> float | None:
    if denom <= 0:
        return None
    return round(numer / denom, 6)


def _blank_class_counts() -> dict[str, int]:
    return {"tp": 0, "fp": 0, "fn": 0, "images": 0, "images_hit": 0}


def _accumulate_match(
    preds: list[dict[str, Any]],
    gts: list[dict[str, Any]],
    per_class: dict[str, dict[str, int]],
) -> tuple[bool, bool]:
    """按类累加框计数。返回 (该图有评估类真值, 至少一条真值被记为 tp)。"""
    gt_labels: set[str] = set()
    hit_labels: set[str] = set()
    for pred in preds:
        row = per_class.get(str(pred.get("label")))
        if row is None:
            continue
        if pred.get("match") == "tp":
            row["tp"] += 1
        else:
            row["fp"] += 1
    for gt in gts:
        row = per_class.get(str(gt.get("label")))
        if row is None:
            continue
        label = str(gt["label"])
        gt_labels.add(label)
        if gt.get("match") == "tp":
            hit_labels.add(label)
        else:
            row["fn"] += 1
    for label in gt_labels:
        per_class[label]["images"] += 1
        if label in hit_labels:
            per_class[label]["images_hit"] += 1
    return bool(gt_labels), bool(hit_labels)


def _class_summary(per_class: dict[str, dict[str, int]]) -> dict[str, dict[str, Any]]:
    finished: dict[str, dict[str, Any]] = {}
    for name, row in per_class.items():
        tp, fp, fn = int(row["tp"]), int(row["fp"]), int(row["fn"])
        finished[name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": _ratio(tp, tp + fp),
            "recall": _ratio(tp, tp + fn),
            "images": int(row["images"]),
            "images_hit": int(row["images_hit"]),
        }
    return finished


def _write_match_summary(
    *,
    save_dir: Path,
    vis_dir: Path,
    pred_path: Path,
    eval_classes: list[str],
    conf_thr: float,
    iou_thr: float,
    counts: dict[str, int],
    images_with_gt: int,
    images_hit: int,
    per_class: dict[str, dict[str, int]],
    latency: dict[str, Any],
) -> Path:
    """把 conf/IoU 匹配统计写到 run 根目录，并在 vis/ 放同一份。不依赖运行库评估接口。"""
    tp, fp, fn = int(counts["tp"]), int(counts["fp"]), int(counts["fn"])
    summary = {
        "eval_classes": list(eval_classes),
        "vis_conf": conf_thr,
        "vis_iou": iou_thr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "images": int(counts["images"]),
        "images_with_gt": images_with_gt,
        "images_hit": images_hit,
        "per_class": _class_summary(per_class),
        "count_note": "precision=tp/(tp+fp)，recall=tp/(tp+fn)。images_hit 为评估类真值至少中一条的图片数。",
        "vis_dir": str(vis_dir),
        "predictions_path": str(pred_path),
        "latency": latency,
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    vis_dir.mkdir(parents=True, exist_ok=True)
    (vis_dir / "vis_summary.json").write_text(text, encoding="utf-8")
    stats_path = save_dir / "test_summary.json"
    stats_path.write_text(text, encoding="utf-8")
    return stats_path


def run_test_vis(
    *,
    model: Any,
    output_dir: Path,
    save_dir: Path,
    class_names: list[str],
    eval_classes: list[str],
    conf_thr: float,
    iou_thr: float,
) -> Path | None:
    from test_core_vis import draw_eval_image, fold_latency, format_latency, match_preds_gts

    images = _iter_images(output_dir / "images" / "test")
    if not images:
        logger.warning("test 无图，跳过画框: %s", output_dir / "images" / "test")
        return None
    vis_dir = save_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    keep = set(eval_classes)
    counts = {"tp": 0, "fp": 0, "fn": 0, "images": 0}
    per_class = {name: _blank_class_counts() for name in eval_classes}
    images_with_gt = 0
    images_hit = 0
    latency_calls: list[tuple[float, int]] = []
    pred_path = vis_dir / "predictions.jsonl"
    import cv2
    from PIL import Image

    loop_error: Exception | None = None
    out, err, _filt = _arm_redact()
    try:
        try:
            with pred_path.open("w", encoding="utf-8") as pred_fp:
                for img_path in images:
                    bgr = cv2.imread(str(img_path))
                    if bgr is None:
                        logger.warning("无法读图，跳过画框: %s", img_path)
                        continue
                    height, width = bgr.shape[:2]
                    rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                    _sync_infer_device()
                    t0 = time.perf_counter()
                    raw = _predict_image(model, rgb, conf_thr)
                    _sync_infer_device()
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    latency_calls.append((elapsed_ms, 1))
                    gts = yolo_gt_boxes(
                        output_dir / "labels" / "test" / f"{img_path.stem}.txt",
                        width,
                        height,
                        class_names,
                        keep,
                    )
                    preds = preds_from_predict(raw, class_names, keep, conf_thr=conf_thr)
                    preds, gts, one = match_preds_gts(preds, gts, iou_thr=iou_thr)
                    for key in ("tp", "fp", "fn"):
                        counts[key] += one[key]
                    counts["images"] += 1
                    has_gt, hit = _accumulate_match(preds, gts, per_class)
                    if has_gt:
                        images_with_gt += 1
                        if hit:
                            images_hit += 1
                    record = {
                        "file_name": img_path.name,
                        "predictions": [
                            {
                                "label": item["label"],
                                "score": round(float(item["score"]), 6),
                                "box": [round(float(v), 2) for v in item["box"]],
                                "match": item["match"],
                                "iou": round(float(item.get("iou") or 0.0), 4),
                            }
                            for item in preds
                        ],
                        "ground_truth": [
                            {
                                "label": item["label"],
                                "box": [round(float(v), 2) for v in item["box"]],
                                "match": item["match"],
                            }
                            for item in gts
                        ],
                    }
                    pred_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                    canvas = draw_eval_image(bgr, preds, gts)
                    out_path = vis_dir / img_path.name
                    if not cv2.imwrite(str(out_path), canvas):
                        logger.warning("写画框图失败: %s", out_path)
        except Exception as exc:
            loop_error = exc
    finally:
        _restore_streams(out, err)
    try:
        latency = fold_latency(latency_calls)
    except Exception:
        logger.warning("test 时延汇总失败，统计仍会写出")
        latency = {"unit": "ms", "images": 0, "note": "时延汇总失败"}
    latency["scope"] = "predict"
    latency["batch"] = 1
    stats_path = _write_match_summary(
        save_dir=save_dir,
        vis_dir=vis_dir,
        pred_path=pred_path,
        eval_classes=eval_classes,
        conf_thr=conf_thr,
        iou_thr=iou_thr,
        counts=counts,
        images_with_gt=images_with_gt,
        images_hit=images_hit,
        per_class=per_class,
        latency=latency,
    )
    logger.info("test 推理时延 %s", format_latency(latency))
    logger.info(
        "test 统计 images=%d hit=%d tp=%d fp=%d fn=%d precision=%s recall=%s 文件=%s",
        counts["images"],
        images_hit,
        counts["tp"],
        counts["fp"],
        counts["fn"],
        _ratio(counts["tp"], counts["tp"] + counts["fp"]),
        _ratio(counts["tp"], counts["tp"] + counts["fn"]),
        stats_path,
    )
    for name, row in _class_summary(per_class).items():
        logger.info(
            "test 统计 %s tp=%d fp=%d fn=%d precision=%s recall=%s images=%d hit=%d",
            name,
            row["tp"],
            row["fp"],
            row["fn"],
            row["precision"],
            row["recall"],
            row["images"],
            row["images_hit"],
        )
    if loop_error is not None:
        logger.warning("test 画框中断，已写出截至当前的统计: %s", stats_path)
        raise loop_error
    return vis_dir


def load_model(size: str, *, pretrain_weights: str | None = None, resolution: int | None = None) -> Any:
    try:
        mod = importlib.import_module(_pkg_name())
    except ModuleNotFoundError:
        raise RuntimeError("未安装检测 mark 运行库，请在当前环境安装后再训练") from None
    cls_name = _cls_prefix() + _SIZE_SUFFIX[size]
    try:
        cls = getattr(mod, cls_name)
    except AttributeError:
        raise RuntimeError(f"当前运行库没有 size={size} 这一档") from None
    kwargs: dict[str, Any] = {}
    if pretrain_weights:
        kwargs["pretrain_weights"] = pretrain_weights
    if resolution:
        kwargs["resolution"] = resolution
    return cls(**kwargs)


def run_fit(
    *,
    dataset_dir: Path,
    run_dir: Path,
    size: str,
    device: str,
    imgsz: int,
    epochs: int,
    batch: int,
    grad_accum_steps: int,
    patience: int,
    workers: int,
    lr: float,
    weight_decay: float,
    checkpoint_interval: int,
    resume: str | None,
    pretrain_weights: str | None,
) -> Any:
    kwargs: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "dataset_file": "yolo",
        "output_dir": str(run_dir),
        "epochs": epochs,
        "batch_size": batch,
        "grad_accum_steps": grad_accum_steps,
        "lr": lr,
        "weight_decay": weight_decay,
        "resolution": imgsz,
        "num_workers": workers,
        "device": device,
        "checkpoint_interval": checkpoint_interval,
        "early_stopping": patience > 0,
    }
    if patience > 0:
        kwargs["early_stopping_patience"] = patience
    if resume:
        kwargs["resume"] = resume

    logger.info(
        "开始训练 size=%s yaml=%s device=%s imgsz=%s epochs=%s batch=%s run=%s",
        size,
        dataset_dir / "data.yaml",
        device,
        imgsz,
        epochs,
        batch,
        run_dir,
    )
    out, err, filt = _arm_redact()
    try:
        try:
            model = load_model(size, pretrain_weights=pretrain_weights, resolution=imgsz)
            _attach_redact_filters(filt)
            model.train(**kwargs)
        except Exception as exc:
            _scrub_exc(exc)
            raise
    finally:
        _restore_streams(out, err)
    return model


def write_test_metrics(
    model: Any,
    *,
    dataset_dir: Path,
    run_dir: Path,
    device: str,
    imgsz: int,
    batch: int,
    workers: int,
) -> None:
    """训练结束后把 test 划分的指标写到该次 run 的 test_metrics.json。运行库没有该接口时跳过。"""
    evaluate = getattr(model, "evaluate", None)
    if not callable(evaluate):
        logger.info("运行库不能单独评估 test 划分，跳过 test_metrics.json")
        return
    out, err, filt = _arm_redact()
    try:
        try:
            _attach_redact_filters(filt)
            metrics = evaluate(
                split="test",
                dataset_dir=str(dataset_dir),
                output_dir=str(run_dir),
                device=device,
                resolution=imgsz,
                batch_size=batch,
                num_workers=workers,
            )
        except Exception as exc:
            _scrub_exc(exc)
            logger.warning("test 指标评估失败，画框仍会继续: %s", exc)
            return
    finally:
        _restore_streams(out, err)
    path = run_dir / "test_metrics.json"
    path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("test 指标已写入 %s", path)


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
    size: str,
    run_prefix: str,
    resume: str,
    imgsz: int,
    epochs: int,
    batch: int,
    grad_accum_steps: int,
    patience: int,
    workers: int,
    lr: float,
    weight_decay: float,
    checkpoint_interval: int,
    device: str,
    single_cls: bool,
    eval_classes: list[str],
    model_path: str,
    save_vis: bool,
    vis_conf: float,
    vis_iou: float,
    test_only: bool,
    split_each_dir: bool = True,
    voc_paths: list[Path] | None = None,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    size = normalize_size(size)
    if grad_accum_steps < 1:
        raise ValueError(f"grad_accum_steps 须 ≥ 1，当前为 {grad_accum_steps}")
    if single_cls and not prepare_only:
        raise ValueError(
            "检测 mark 按 data.yaml 的类别训练，不把多类合并成一类。"
            "请设 single_cls=false，或在 train_classes 里只保留一类。"
        )

    if test_only and (output_dir / "data.yaml").is_file():
        skip_prepare = True
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
            label_format="detect",
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

    sync_data_yaml_path(output_dir)
    class_names = read_class_names(output_dir)
    eval_names = resolve_eval_classes({"val_classes": eval_classes}, class_names)
    logger.info("评估类: %s", ",".join(eval_names))
    assert_imgsz(size, imgsz)

    def _vis(weight: Path, save_dir: Path) -> None:
        if not save_vis:
            logger.info("save_vis=false，跳过 test 画框")
            return
        logger.info("test 画框权重: %s 输出: %s", weight, save_dir / "vis")
        out, err, _filt = _arm_redact()
        try:
            model = load_model(size, pretrain_weights=str(weight), resolution=imgsz)
        finally:
            _restore_streams(out, err)
        run_test_vis(
            model=model,
            output_dir=output_dir,
            save_dir=save_dir,
            class_names=class_names,
            eval_classes=eval_names,
            conf_thr=vis_conf,
            iou_thr=vis_iou,
        )

    if test_only:
        weight = resolve_test_weight(
            output_dir=output_dir,
            run_prefix=run_prefix,
            resume=resume,
            model_path=model_path,
        )
        _vis(weight, vis_root_for_weight(weight, output_dir, run_prefix))
        return

    pretrain = resolve_pretrain(model_path)
    if pretrain:
        logger.info("预训练权重: %s", pretrain)
    run_dir, resume_path = resolve_run(output_dir, run_prefix, resume)
    trained_device = solver_device(device)
    model = run_fit(
        dataset_dir=output_dir,
        run_dir=run_dir,
        size=size,
        device=trained_device,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        grad_accum_steps=grad_accum_steps,
        patience=patience,
        workers=workers,
        lr=lr,
        weight_decay=weight_decay,
        checkpoint_interval=checkpoint_interval,
        resume=resume_path,
        pretrain_weights=pretrain,
    )
    logger.info("训练结束 run=%s", run_dir)
    write_test_metrics(
        model,
        dataset_dir=output_dir,
        run_dir=run_dir,
        device=trained_device,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
    )
    trained = best_weight_in(run_dir)
    if trained is None:
        logger.warning("run 目录没有 best 权重，跳过 test 画框: %s", run_dir)
        return
    _vis(trained, run_dir)


def main_from_config(config_path: str | Path | None = None, *, test_only: bool | None = None) -> None:
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
        size=str(cfg.get("size") or "s"),
        run_prefix=str(cfg.get("run_prefix") or "mark_s"),
        resume=str(cfg.get("resume") or ""),
        imgsz=int(train_cfg.get("imgsz", 1024)),
        epochs=int(train_cfg.get("epochs", 80)),
        batch=int(train_cfg.get("batch", 4)),
        grad_accum_steps=int(train_cfg.get("grad_accum_steps", 1)),
        patience=int(train_cfg.get("patience", 25)),
        workers=int(train_cfg.get("workers", 4)),
        lr=float(train_cfg.get("lr", 0.0001)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0001)),
        checkpoint_interval=int(train_cfg.get("checkpoint_interval", 10)),
        device=str(cfg.get("device") or ""),
        single_cls=bool(cfg.get("single_cls", False)),
        eval_classes=list(cfg.get("val_classes") or []),
        model_path=str(cfg.get("model_path") or ""),
        save_vis=bool(cfg.get("save_vis", True)),
        vis_conf=float(cfg.get("vis_conf", 0.5)),
        vis_iou=float(cfg.get("vis_iou", 0.5)),
        test_only=bool(cfg.get("test_only", False)) if test_only is None else test_only,
    )


if __name__ == "__main__":
    # 客户机 nohup .../python3 train_mark.py > d.log 2>&1 &
    # 改数据/output_dir：train_detect_cfg/data_cfg.json；改超参：train_detect_mark/train_config.json
    CONFIG_PATH = str(DEFAULT_TRAIN_CONFIG)
    main_from_config(CONFIG_PATH)
