#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检测 core 评估可视化：TP/FP/FN 画框与预测置信度记录（仅 scanly/train 内使用）。"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# BGR
_COLOR_TP = (0, 180, 0)
_COLOR_FP = (0, 0, 220)
_COLOR_FN = (0, 140, 255)


def _cv2():
    import cv2

    return cv2


def unwrap_dataset(dataset: Any) -> Any:
    ds = dataset
    while hasattr(ds, "dataset"):
        ds = ds.dataset
    return ds


def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def coco_xywh_to_xyxy(bbox: list[float]) -> list[float]:
    x, y, w, h = [float(v) for v in bbox]
    return [x, y, x + w, y + h]


def label_name(label_id: int, class_names: list[str]) -> str:
    if 0 <= label_id < len(class_names):
        return class_names[label_id]
    return str(label_id)


def load_gt_boxes(
    dataset: Any,
    image_id: int,
    class_names: list[str],
) -> tuple[Path, list[dict[str, Any]]]:
    """从 COCO GT 取原图像素框；返回 (image_path, gt_list)。"""
    info = dataset.coco.loadImgs(int(image_id))[0]
    img_path = Path(dataset.img_folder) / info["file_name"]
    ann_ids = dataset.coco.getAnnIds(imgIds=[int(image_id)])
    anns = dataset.coco.loadAnns(ann_ids)
    gts: list[dict[str, Any]] = []
    for ann in anns:
        if int(ann.get("iscrowd", 0)) != 0:
            continue
        cat_id = int(ann["category_id"])
        gts.append(
            {
                "label_id": cat_id,
                "label": label_name(cat_id, class_names),
                "box": coco_xywh_to_xyxy(ann["bbox"]),
            }
        )
    return img_path, gts


def preds_from_output(
    output: dict[str, Any],
    class_names: list[str],
    *,
    conf_thr: float,
) -> list[dict[str, Any]]:
    labels = np.asarray(output["labels"].detach().cpu().numpy()).reshape(-1)
    boxes = np.asarray(output["boxes"].detach().cpu().numpy()).reshape(-1, 4)
    scores = np.asarray(output["scores"].detach().cpu().numpy()).reshape(-1)
    preds: list[dict[str, Any]] = []
    for lab, box, score in zip(labels, boxes, scores):
        sc = float(score)
        if sc < conf_thr:
            continue
        lid = int(lab)
        preds.append(
            {
                "label_id": lid,
                "label": label_name(lid, class_names),
                "score": sc,
                "box": [float(x) for x in box.tolist()],
            }
        )
    preds.sort(key=lambda p: p["score"], reverse=True)
    return preds


def match_preds_gts(
    preds: list[dict[str, Any]],
    gts: list[dict[str, Any]],
    *,
    iou_thr: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """贪心匹配；返回带 match 字段的 preds/gts 与计数。"""
    gt_used = [False] * len(gts)
    matched_preds: list[dict[str, Any]] = []
    counts = {"tp": 0, "fp": 0, "fn": 0}

    for pred in preds:
        best_i = -1
        best_iou = 0.0
        for i, gt in enumerate(gts):
            if gt_used[i]:
                continue
            if int(gt["label_id"]) != int(pred["label_id"]):
                continue
            iou = box_iou_xyxy(np.asarray(pred["box"]), np.asarray(gt["box"]))
            if iou >= iou_thr and iou > best_iou:
                best_iou = iou
                best_i = i
        item = dict(pred)
        if best_i >= 0:
            gt_used[best_i] = True
            item["match"] = "tp"
            item["iou"] = round(best_iou, 4)
            counts["tp"] += 1
        else:
            item["match"] = "fp"
            item["iou"] = 0.0
            counts["fp"] += 1
        matched_preds.append(item)

    matched_gts: list[dict[str, Any]] = []
    for i, gt in enumerate(gts):
        item = dict(gt)
        if gt_used[i]:
            item["match"] = "tp"
        else:
            item["match"] = "fn"
            counts["fn"] += 1
        matched_gts.append(item)
    return matched_preds, matched_gts, counts


def _label_rect(x: int, baseline_y: int, tw: int, th: int, baseline: int) -> tuple[int, int, int, int]:
    return (x, baseline_y - th - 2, x + tw + 2, baseline_y + baseline)


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def resolve_label_pos(
    *,
    x: int,
    y: int,
    tw: int,
    th: int,
    baseline: int,
    image_w: int,
    image_h: int,
    occupied: list[tuple[int, int, int, int]],
) -> tuple[int, int, tuple[int, int, int, int]]:
    """把标签放到锚点附近，并避开已经占用的矩形。优先下移，放不下再右移。"""
    gap = 2
    max_x = max(0, image_w - tw - 2)
    x = min(max(0, x), max_x)
    prefer_y = max(th + 2, y)
    floor_y = max(th + 2, image_h - baseline)
    baseline_y = min(prefer_y, floor_y)

    for _ in range(64):
        rect = _label_rect(x, baseline_y, tw, th, baseline)
        if rect[3] > image_h or rect[1] < 0:
            break
        blocker = next((item for item in occupied if _rects_overlap(rect, item)), None)
        if blocker is None:
            return x, baseline_y, rect
        moved = blocker[3] + gap + th + 2
        if moved + baseline <= image_h:
            baseline_y = moved
            continue
        nxt = min(blocker[2] + gap, max_x)
        if nxt <= x:
            break
        x = nxt
        baseline_y = min(prefer_y, floor_y)

    rect = _label_rect(x, min(prefer_y, floor_y), tw, th, baseline)
    return x, min(prefer_y, floor_y), rect


def _put_label(
    canvas: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int],
    occupied: list[tuple[int, int, int, int]],
) -> None:
    cv2 = _cv2()
    x, y = org
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    image_h, image_w = canvas.shape[:2]
    x, y0, rect = resolve_label_pos(
        x=x,
        y=y,
        tw=tw,
        th=th,
        baseline=baseline,
        image_w=image_w,
        image_h=image_h,
        occupied=occupied,
    )
    occupied.append(rect)
    cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), color, thickness=-1)
    cv2.putText(
        canvas,
        text,
        (x + 1, y0),
        font,
        scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )


def draw_eval_image(
    image_bgr: np.ndarray,
    preds: list[dict[str, Any]],
    gts: list[dict[str, Any]],
) -> np.ndarray:
    cv2 = _cv2()
    canvas = image_bgr.copy()
    pending: list[tuple[str, tuple[int, int], tuple[int, int, int]]] = []
    for gt in gts:
        if gt.get("match") != "fn":
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in gt["box"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), _COLOR_FN, 2)
        pending.append((f"FN {gt['label']}", (x1, y1 - 2), _COLOR_FN))
    for pred in preds:
        match = pred.get("match", "fp")
        color = _COLOR_TP if match == "tp" else _COLOR_FP
        tag = "TP" if match == "tp" else "FP"
        x1, y1, x2, y2 = [int(round(v)) for v in pred["box"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        score = float(pred.get("score", 0.0))
        pending.append((f"{tag} {pred['label']} {score:.2f}", (x1, y1 - 2), color))
    occupied: list[tuple[int, int, int, int]] = []
    for text, org, color in pending:
        _put_label(canvas, text, org, color, occupied)
    return canvas


class VisWriter:
    """评估过程中按图写 vis / predictions.jsonl，结束时写 vis_summary。"""

    def __init__(
        self,
        save_dir: Path,
        class_names: list[str],
        *,
        conf_thr: float,
        iou_thr: float,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.vis_dir = self.save_dir / "vis"
        self.predictions_path = self.save_dir / "predictions.jsonl"
        self.summary_path = self.save_dir / "vis_summary.json"
        self.class_names = list(class_names)
        self.conf_thr = float(conf_thr)
        self.iou_thr = float(iou_thr)
        self.counts = {"tp": 0, "fp": 0, "fn": 0, "images": 0}
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        self._pred_fp = self.predictions_path.open("w", encoding="utf-8")

    def process_batch(
        self,
        dataset: Any,
        targets: list[dict[str, Any]],
        results: list[dict[str, Any]],
    ) -> None:
        for target, output in zip(targets, results):
            image_id = int(target["image_id"].item())
            img_path, gts_raw = load_gt_boxes(dataset, image_id, self.class_names)
            preds_raw = preds_from_output(
                output, self.class_names, conf_thr=self.conf_thr
            )
            preds, gts, counts = match_preds_gts(
                preds_raw, gts_raw, iou_thr=self.iou_thr
            )
            for k in ("tp", "fp", "fn"):
                self.counts[k] += counts[k]
            self.counts["images"] += 1

            record = {
                "image_id": image_id,
                "file_name": img_path.name,
                "predictions": [
                    {
                        "label": p["label"],
                        "score": round(float(p["score"]), 6),
                        "box": [round(float(x), 2) for x in p["box"]],
                        "match": p["match"],
                        "iou": p.get("iou", 0.0),
                    }
                    for p in preds
                ],
                "ground_truth": [
                    {
                        "label": g["label"],
                        "box": [round(float(x), 2) for x in g["box"]],
                        "match": g["match"],
                    }
                    for g in gts
                ],
            }
            self._pred_fp.write(json.dumps(record, ensure_ascii=False) + "\n")

            image = _cv2().imread(str(img_path))
            if image is None:
                logger.warning("无法读图，跳过画框: %s", img_path)
                continue
            canvas = draw_eval_image(image, preds, gts)
            out_path = self.vis_dir / img_path.name
            if not _cv2().imwrite(str(out_path), canvas):
                logger.warning("写画框图失败: %s", out_path)

    def close(self) -> dict[str, Any]:
        self._pred_fp.close()
        summary = {
            "vis_conf": self.conf_thr,
            "vis_iou": self.iou_thr,
            "images": self.counts["images"],
            "tp": self.counts["tp"],
            "fp": self.counts["fp"],
            "fn": self.counts["fn"],
            "vis_dir": str(self.vis_dir),
            "predictions_path": str(self.predictions_path),
        }
        self.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info(
            "可视化完成 images=%d tp=%d fp=%d fn=%d vis=%s preds=%s",
            summary["images"],
            summary["tp"],
            summary["fp"],
            summary["fn"],
            self.vis_dir,
            self.predictions_path,
        )
        return summary


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    weight = rank - lo
    return sorted_vals[lo] * (1.0 - weight) + sorted_vals[hi] * weight


def fold_latency(calls: list[tuple[float, int]]) -> dict[str, Any]:
    """把多次推理调用收成每张图的毫秒。调用多于一次时，第一次整批不计入。"""
    if len(calls) > 1:
        warmup_excluded = int(calls[0][1])
        kept = calls[1:]
    else:
        warmup_excluded = 0
        kept = list(calls)
    samples: list[float] = []
    for per_ms, count in kept:
        samples.extend([float(per_ms)] * max(0, int(count)))
    if not samples:
        return {
            "unit": "ms",
            "images": 0,
            "warmup_excluded": warmup_excluded,
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "min_ms": None,
            "max_ms": None,
            "note": "每张图的推理毫秒；调用多于一次时不含第一次",
        }
    ordered = sorted(samples)
    n = len(ordered)
    return {
        "unit": "ms",
        "images": n,
        "warmup_excluded": warmup_excluded,
        "mean_ms": round(sum(ordered) / n, 3),
        "p50_ms": round(_percentile(ordered, 50), 3),
        "p95_ms": round(_percentile(ordered, 95), 3),
        "min_ms": round(ordered[0], 3),
        "max_ms": round(ordered[-1], 3),
        "note": "每张图的推理毫秒；调用多于一次时不含第一次",
    }


def format_latency(summary: dict[str, Any]) -> str:
    return (
        "images=%s warmup_excluded=%s mean=%sms p50=%sms p95=%sms min=%sms max=%sms"
        % (
            summary.get("images"),
            summary.get("warmup_excluded"),
            summary.get("mean_ms"),
            summary.get("p50_ms"),
            summary.get("p95_ms"),
            summary.get("min_ms"),
            summary.get("max_ms"),
        )
    )
