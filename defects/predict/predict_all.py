#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Detail  : 封边缺陷统一推理：YOLO 检测 + 方案启停/置信度/覆膜过滤 + NG/贴标/排出判定。

from __future__ import annotations

import logging
import threading
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from labels import cn_name_of
from scheme import judge_box, load_json, merge_scheme, save_json, scheme_public_view, scheme_storage_dict

logger = logging.getLogger(__name__)

_PREDICT_DIR = Path(__file__).resolve().parent
DEFAULT_PREDICT_JSON = _PREDICT_DIR / "config" / "predict.json"
DEFAULT_SCHEME_JSON = _PREDICT_DIR / "config" / "scheme.json"
DEFAULT_MODELS_DIR = _PREDICT_DIR / "models"


def resolve_model_path(raw: str | Path | None) -> str:
    """
    解析模型路径：空串保持空；绝对路径原样；相对路径相对 models/。
    例：cam2-0819.pt → <predict>/models/cam2-0819.pt
    """
    text = str(raw or "").strip()
    if not text:
        return ""
    path = Path(text).expanduser()
    if path.is_absolute():
        return str(path)
    return str((DEFAULT_MODELS_DIR / path).resolve())


def xyxy_to_obb(gx1: float, gy1: float, gx2: float, gy2: float) -> dict[str, Any]:
    """轴对齐框视为 angle=0 的特殊 OBB（tl,tr,br,bl）。"""
    left = float(min(gx1, gx2))
    right = float(max(gx1, gx2))
    top = float(min(gy1, gy2))
    bottom = float(max(gy1, gy2))
    width = max(right - left, 0.0)
    height = max(bottom - top, 0.0)
    return {
        "points": [
            [int(round(left)), int(round(top))],
            [int(round(right)), int(round(top))],
            [int(round(right)), int(round(bottom))],
            [int(round(left)), int(round(bottom))],
        ],
        "cx": round((left + right) / 2.0, 2),
        "cy": round((top + bottom) / 2.0, 2),
        "width": round(width, 2),
        "height": round(height, 2),
        "angle": 0.0,
    }

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover
    Image = None
    ImageDraw = None
    ImageFont = None


def load_image_bgr(path: str | Path) -> np.ndarray:
    p = Path(str(path))
    if not p.is_file():
        raise ValueError(f"图片不存在: {p}")
    data = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise ValueError(f"无法解码图片: {p}")
    return img


def _axis_starts(size: int, clip: int, overlap: int) -> list[int]:
    if clip <= 0 or size <= clip:
        return [0]
    step = max(int(clip) - int(overlap), 1)
    last = size - clip
    starts = list(range(0, last + 1, step))
    if starts[-1] != last:
        starts.append(last)
    return starts


def iter_tiles(
    height: int, width: int, clip_size: int, overlap_size: int
) -> list[tuple[int, int, int, int]]:
    if clip_size <= 0:
        return [(0, 0, width, height)]
    tiles: list[tuple[int, int, int, int]] = []
    clip_w = min(int(clip_size), width)
    clip_h = min(int(clip_size), height)
    for y0 in _axis_starts(height, clip_h, overlap_size):
        for x0 in _axis_starts(width, clip_w, overlap_size):
            tiles.append((x0, y0, x0 + clip_w, y0 + clip_h))
    return tiles


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _as_float_list(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        out: list[float] = []
        for item in value:
            if isinstance(item, (list, tuple)):
                out.extend(float(x) for x in item)
            else:
                out.append(float(item))
        return out
    return [float(value)]


def _parse_detect_pred(
    pred: Any, names: dict[int, str], ox: int, oy: int
) -> list[dict[str, Any]]:
    boxes = getattr(pred, "boxes", None)
    if boxes is None:
        return []
    out: list[dict[str, Any]] = []
    for box in boxes:
        xyxy = _as_float_list(box.xyxy[0])
        if len(xyxy) < 4:
            continue
        cls_id = int(box.cls[0])
        gx1 = float(xyxy[0]) + ox
        gy1 = float(xyxy[1]) + oy
        gx2 = float(xyxy[2]) + ox
        gy2 = float(xyxy[3]) + oy
        out.append(
            {
                "name": names.get(cls_id, str(cls_id)),
                "score": float(box.conf[0]),
                "xyxy": [gx1, gy1, gx2, gy2],
                # bbox 当作特殊 OBB（angle=0）统一输出
                "obb": xyxy_to_obb(gx1, gy1, gx2, gy2),
            }
        )
    return out


def _parse_obb_pred(
    pred: Any, names: dict[int, str], ox: int, oy: int
) -> list[dict[str, Any]]:
    obbs = getattr(pred, "obb", None)
    if obbs is None:
        return []
    out: list[dict[str, Any]] = []
    for box in obbs:
        cls_id = int(box.cls[0])
        score = float(box.conf[0])
        xyxy = _as_float_list(getattr(box, "xyxy")[0])
        corners = _as_float_list(getattr(box, "xyxyxyxy")[0])
        if len(corners) < 8:
            continue
        pts = []
        xs: list[float] = []
        ys: list[float] = []
        for i in range(0, 8, 2):
            px = float(corners[i]) + ox
            py = float(corners[i + 1]) + oy
            pts.append([int(round(px)), int(round(py))])
            xs.append(px)
            ys.append(py)
        if len(xyxy) >= 4:
            gx1, gy1, gx2, gy2 = (
                float(xyxy[0]) + ox,
                float(xyxy[1]) + oy,
                float(xyxy[2]) + ox,
                float(xyxy[3]) + oy,
            )
        else:
            gx1, gx2 = min(xs), max(xs)
            gy1, gy2 = min(ys), max(ys)
        xywhr = _as_float_list(getattr(box, "xywhr")[0])
        cx = float(xywhr[0]) + ox if len(xywhr) >= 5 else (gx1 + gx2) / 2.0
        cy = float(xywhr[1]) + oy if len(xywhr) >= 5 else (gy1 + gy2) / 2.0
        bw = float(xywhr[2]) if len(xywhr) >= 5 else max(gx2 - gx1, 0.0)
        bh = float(xywhr[3]) if len(xywhr) >= 5 else max(gy2 - gy1, 0.0)
        angle = float(xywhr[4]) if len(xywhr) >= 5 else 0.0
        out.append(
            {
                "name": names.get(cls_id, str(cls_id)),
                "score": score,
                "xyxy": [gx1, gy1, gx2, gy2],
                "obb": {
                    "points": pts,
                    "cx": round(cx, 2),
                    "cy": round(cy, 2),
                    "width": round(bw, 2),
                    "height": round(bh, 2),
                    "angle": round(angle, 6),
                },
            }
        )
    return out


def nms_boxes(
    boxes: list[dict[str, Any]], iou_thresh: float
) -> list[dict[str, Any]]:
    if not boxes:
        return []
    order = sorted(range(len(boxes)), key=lambda i: float(boxes[i]["score"]), reverse=True)
    keep: list[int] = []
    while order:
        i = order.pop(0)
        keep.append(i)
        remain = []
        ai = np.array(boxes[i]["xyxy"], dtype=float)
        for j in order:
            bj = np.array(boxes[j]["xyxy"], dtype=float)
            same = str(boxes[i]["name"]) == str(boxes[j]["name"])
            if same and _iou_xyxy(ai, bj) >= iou_thresh:
                continue
            remain.append(j)
        order = remain
    return [boxes[i] for i in keep]


def _resolve_cjk_font(size: int = 18) -> Any:
    """解析可绘制中文的字体；TTC 尝试多子字体；失败时告警并回退。"""
    if ImageFont is None:
        return None

    candidates = [
        # macOS（新系统常无 PingFang.ttc 单文件路径）
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        # Windows
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        # Linux / Ubuntu 常见中文字体
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/usr/share/fonts/truetype/arphic/ukai.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
    ]
    # 额外扫描常见目录，兼容 fonts-noto-cjk / fonts-wqy 包安装布局
    for root in (
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path.home() / ".local/share/fonts",
    ):
        if not root.is_dir():
            continue
        for pattern in (
            "**/NotoSansCJK*Regular*.otf",
            "**/NotoSansCJK*Regular*.ttc",
            "**/NotoSansCJKsc*.otf",
            "**/wqy-microhei.*",
            "**/wqy-zenhei.*",
            "**/DroidSansFallback*.ttf",
        ):
            for hit in root.glob(pattern):
                text = str(hit)
                if text not in candidates:
                    candidates.append(text)
    sample = "短带胶缝"

    def _can_draw_cjk(font: Any) -> bool:
        try:
            if hasattr(font, "getbbox"):
                box = font.getbbox(sample)
                return bool(box) and (box[2] - box[0]) >= len(sample) * max(int(size * 0.35), 4)
            mask = font.getmask(sample)
            return bool(getattr(mask, "size", (0, 0))[0])
        except Exception:
            return False

    for path in candidates:
        if not Path(path).is_file():
            continue
        # .ttf/.otf 通常无 index；.ttc 尝试多个 face
        indexes = range(0, 8) if path.lower().endswith(".ttc") else (0,)
        for idx in indexes:
            try:
                font = ImageFont.truetype(path, size=size, index=int(idx))
            except OSError:
                break
            except Exception:
                continue
            if _can_draw_cjk(font):
                logger.info("Gradio 画字字体: %s index=%s", path, idx)
                return font

    logger.warning(
        "未找到可用中文字体，结果图中文可能显示为方框；"
        "请安装 Noto Sans CJK / 微软雅黑 / 文泉驿等字体。"
        " Ubuntu 可: sudo apt install fonts-wqy-microhei fonts-noto-cjk"
    )
    return ImageFont.load_default() if ImageFont else None


_CJK_FONT_CACHE: dict[int, Any] = {}


def _cjk_font(size: int = 18) -> Any:
    font = _CJK_FONT_CACHE.get(size)
    if font is None:
        font = _resolve_cjk_font(size)
        _CJK_FONT_CACHE[size] = font
    return font


def draw_results(image_bgr: np.ndarray, results: list[dict[str, Any]]) -> np.ndarray:
    canvas = image_bgr.copy()
    if Image is None or ImageDraw is None:
        for item in results:
            color = (0, 0, 255) if item.get("ng") else (0, 180, 0)
            pts = _obb_points(item)
            if pts is not None:
                cv2.polylines(canvas, [pts], True, color, 2)
            else:
                loc = item.get("location") or {}
                x1 = int(loc.get("left") or 0)
                y1 = int(loc.get("top") or 0)
                x2 = x1 + int(loc.get("width") or 0)
                y2 = y1 + int(loc.get("height") or 0)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        return canvas

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    font = _cjk_font(18)
    for item in results:
        loc = item.get("location") or {}
        x1 = int(loc.get("left") or 0)
        y1 = int(loc.get("top") or 0)
        x2 = x1 + int(loc.get("width") or 0)
        y2 = y1 + int(loc.get("height") or 0)
        color = (220, 40, 40) if item.get("ng") else (40, 160, 70)
        pts = _obb_points(item)
        if pts is not None:
            seq = [tuple(int(v) for v in p) for p in pts.tolist()]
            draw.line(seq + [seq[0]], fill=color, width=2)
            tx, ty = seq[0]
        else:
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            tx, ty = x1, y1
        cn = str(item.get("cn_name") or "")
        name = str(item.get("name") or "")
        flags = []
        if item.get("ng"):
            flags.append("NG")
        if item.get("need_label"):
            flags.append("标")
        if item.get("need_discharge"):
            flags.append("排")
        label = f"{cn or name} {float(item.get('score') or 0):.2f}"
        if flags:
            label = f"{label} [{'+'.join(flags)}]"
        # 按实际文字宽度铺底，避免中文标签被裁切
        try:
            if hasattr(font, "getbbox"):
                lb = font.getbbox(label)
                text_w = max(lb[2] - lb[0], 40)
                text_h = max(lb[3] - lb[1], 16)
            else:
                text_w = max(10 * len(label), 40)
                text_h = 18
        except Exception:
            text_w = max(10 * len(label), 40)
            text_h = 18
        ty = max(ty - text_h - 2, 0)
        draw.rectangle([tx, ty, tx + text_w + 6, ty + text_h + 4], fill=color)
        draw.text((tx + 2, ty + 1), label, fill=(255, 255, 255), font=font)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def _obb_points(item: dict[str, Any]) -> np.ndarray | None:
    raw = (item.get("obb") or {}).get("points")
    if not isinstance(raw, list) or len(raw) < 4:
        return None
    pts = []
    for p in raw[:4]:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            return None
        pts.append([int(p[0]), int(p[1])])
    return np.array(pts, dtype=np.int32)


def normalize_infer_type(raw: Any, *, default: str = "obb") -> str:
    if raw is None or str(raw).strip() == "":
        text = default
    else:
        text = str(raw).strip().lower()
    if text in {"obb", "oriented", "rotate", "rotated"}:
        return "obb"
    if text in {"detect", "det", "bbox"}:
        return "detect"
    raise ValueError(f"不支持的推理类型: {raw}")


def _parse_class_names(raw: Any) -> list[str]:
    if raw is None:
        return ["defect"]
    if isinstance(raw, str):
        text = raw.strip()
        return [text] if text else ["defect"]
    if isinstance(raw, (list, tuple)):
        names = [str(x).strip() for x in raw if str(x).strip()]
        return names or ["defect"]
    return ["defect"]


def _is_rtdetr_onnx_inputs(input_names: list[str]) -> bool:
    names = {str(n) for n in input_names}
    return "images" in names and "orig_target_sizes" in names


def probe_onnx_kind(model_path: str | Path) -> str:
    """返回 ``rtdetr`` 或 ``yolo``（其它 ONNX 默认按 YOLO/Ultralytics 尝试）。"""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "加载 .onnx 需要 onnxruntime，请安装 scanly/defects/predict/requirements.txt"
        ) from exc
    sess = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    names = [str(i.name) for i in sess.get_inputs()]
    if _is_rtdetr_onnx_inputs(names):
        return "rtdetr"
    return "yolo"


class RtdetrOnnxSession:
    """RT-DETRv2 导出 ONNX：images + orig_target_sizes → labels/boxes/scores。"""

    def __init__(
        self,
        model_path: str | Path,
        *,
        class_names: list[str],
        imgsz: int = 0,
        device: str | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "RT-DETRv2 ONNX 需要 onnxruntime，请安装 "
                "scanly/defects/predict/requirements.txt"
            ) from exc
        self.model_path = str(Path(model_path).resolve())
        self.names = {i: str(n) for i, n in enumerate(class_names)}
        self.device = str(device or "cpu")
        providers = ["CPUExecutionProvider"]
        avail = list(ort.get_available_providers())
        if self.device.startswith("cuda") and "CUDAExecutionProvider" in avail:
            idx = 0
            if ":" in self.device:
                try:
                    idx = int(self.device.split(":")[-1])
                except ValueError:
                    idx = 0
            providers = [
                ("CUDAExecutionProvider", {"device_id": idx}),
                "CPUExecutionProvider",
            ]
        elif self.device.startswith("cuda"):
            logger.warning(
                "RT-DETR ONNX 请求 CUDA 但无 CUDAExecutionProvider，回退 CPU: %s",
                avail,
            )
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            self.model_path, sess_options=so, providers=providers
        )
        inputs = {str(i.name): i for i in self.session.get_inputs()}
        if not _is_rtdetr_onnx_inputs(list(inputs)):
            raise ValueError(
                f"不是 RT-DETRv2 ONNX（需要 images + orig_target_sizes）: {self.model_path}"
            )
        self._image_name = "images"
        self._size_name = "orig_target_sizes"
        shape = inputs[self._image_name].shape
        side = 0
        if shape is not None and len(shape) >= 4:
            for dim in (shape[2], shape[3]):
                try:
                    side = int(dim)
                except (TypeError, ValueError):
                    side = 0
                if side > 0:
                    break
        cfg_imgsz = int(imgsz or 0)
        self.imgsz = cfg_imgsz if cfg_imgsz > 0 else (side if side > 0 else 1024)
        outs = [str(o.name) for o in self.session.get_outputs()]
        if all(n in outs for n in ("labels", "boxes", "scores")):
            self._out_names = ["labels", "boxes", "scores"]
        elif len(outs) >= 3:
            self._out_names = outs[:3]
        else:
            raise ValueError(f"RT-DETRv2 ONNX 输出异常: {outs}")
        logger.info(
            "加载 RT-DETRv2 ONNX: path=%s providers=%s imgsz=%s names=%s",
            self.model_path,
            self.session.get_providers(),
            self.imgsz,
            self.names,
        )

    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        work = image_bgr
        if work.ndim == 2:
            work = cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)
        elif work.ndim == 3 and work.shape[2] == 1:
            work = cv2.cvtColor(work[:, :, 0], cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
        side = max(1, int(self.imgsz))
        if rgb.shape[0] != side or rgb.shape[1] != side:
            rgb = cv2.resize(rgb, (side, side), interpolation=cv2.INTER_LINEAR)
        ten = np.ascontiguousarray(rgb).transpose(2, 0, 1).astype(np.float32) / 255.0
        return ten[None, ...]

    def predict_tile(
        self, tile_bgr: np.ndarray, *, conf: float, ox: int, oy: int
    ) -> list[dict[str, Any]]:
        h, w = tile_bgr.shape[:2]
        images = self._preprocess(tile_bgr)
        # 与训练评估一致：orig_target_sizes 为原图宽高，框坐标落在该像素系
        sizes = np.asarray([[int(w), int(h)]], dtype=np.int64)
        labels, boxes, scores = self.session.run(
            self._out_names,
            {self._image_name: images, self._size_name: sizes},
        )
        labels_a = np.asarray(labels)
        boxes_a = np.asarray(boxes)
        scores_a = np.asarray(scores)
        if labels_a.ndim == 1:
            labels_a = labels_a[None, ...]
            boxes_a = boxes_a[None, ...]
            scores_a = scores_a[None, ...]
        out: list[dict[str, Any]] = []
        lab = labels_a[0]
        box = boxes_a[0]
        sco = scores_a[0]
        n = min(len(lab), len(box), len(sco))
        for i in range(n):
            score = float(sco[i])
            if score < conf:
                continue
            xyxy = np.asarray(box[i], dtype=float).reshape(-1)
            if xyxy.size < 4:
                continue
            gx1 = float(xyxy[0]) + ox
            gy1 = float(xyxy[1]) + oy
            gx2 = float(xyxy[2]) + ox
            gy2 = float(xyxy[3]) + oy
            cls_id = int(lab[i])
            out.append(
                {
                    "name": self.names.get(cls_id, str(cls_id)),
                    "score": score,
                    "xyxy": [gx1, gy1, gx2, gy2],
                    "obb": xyxy_to_obb(gx1, gy1, gx2, gy2),
                }
            )
        return out


class DefectPredictAll:
    """加载一次检测/OBB 模型，重复推理。"""

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        scheme_path: str | Path | None = None,
        predict_cfg: dict[str, Any] | None = None,
        device: str | None = None,
    ) -> None:
        cfg = dict(predict_cfg or {})
        if not cfg and DEFAULT_PREDICT_JSON.is_file():
            cfg = load_json(DEFAULT_PREDICT_JSON)
        self.predict_cfg = cfg
        self.clip_size = int(cfg.get("clip_size") or 0)
        self.overlap_size = int(cfg.get("overlap_size") or 120)
        self.conf = float(cfg.get("conf") or 0.25)
        self.iou = float(cfg.get("iou") or 0.45)
        self.default_mm_per_px = float(cfg.get("mm_per_px") or 0.03)
        self.imgsz = int(cfg.get("imgsz") or 0)
        self.class_names = _parse_class_names(cfg.get("class_names"))
        self.device = device if device is not None else cfg.get("device")
        scheme_file = Path(scheme_path) if scheme_path else DEFAULT_SCHEME_JSON
        self.scheme_path = scheme_file
        self.base_scheme = load_json(scheme_file)
        self._scheme_lock = threading.Lock()
        detect_path = resolve_model_path(model_path or cfg.get("model_path") or "")
        obb_path = resolve_model_path(cfg.get("obb_model_path") or "")
        self.default_infer_type = normalize_infer_type(
            cfg.get("infer_type"), default="obb"
        )
        # RT-DETRv2 ONNX 仅 detect：不要把唯一路径误当成 OBB
        if (
            self.default_infer_type == "obb"
            and not obb_path
            and detect_path
            and str(detect_path).lower().endswith(".onnx")
        ):
            try:
                if probe_onnx_kind(detect_path) == "rtdetr":
                    logger.warning(
                        "RT-DETRv2 ONNX 仅支持 detect，已将默认 infer_type 从 obb 改为 detect: %s",
                        detect_path,
                    )
                    self.default_infer_type = "detect"
            except Exception as exc:
                logger.warning("探测 ONNX 类型失败，仍按配置处理: %s", exc)
        if self.default_infer_type == "obb" and not obb_path and detect_path:
            obb_path = detect_path
            detect_path = ""
        self.model_path = detect_path
        self.obb_model_path = obb_path
        self._models: dict[str, Any] = {}
        self._names: dict[str, dict[int, str]] = {"detect": {}, "obb": {}}
        self._infer_lock = threading.Lock()
        if detect_path:
            self._load_model(detect_path, "detect")
        if obb_path:
            self._load_model(obb_path, "obb")

    def _load_model(self, model_path: str, infer_type: str) -> None:
        p = Path(model_path)
        if not p.is_file():
            logger.warning("%s 模型文件不存在，服务可启动但该类型预测未就绪: %s", infer_type, p)
            return
        suffix = p.suffix.lower()
        if suffix == ".onnx":
            try:
                kind = probe_onnx_kind(p)
            except Exception as exc:
                logger.warning("无法探测 ONNX 类型，跳过加载 %s: %s", p, exc)
                return
            if kind == "rtdetr":
                if infer_type == "obb":
                    logger.warning(
                        "RT-DETRv2 ONNX 不能作为 OBB 权重，改注册为 detect: %s", p
                    )
                    infer_type = "detect"
                    if not self.model_path:
                        self.model_path = str(p)
                    if self.obb_model_path == str(p):
                        self.obb_model_path = ""
                    if self.default_infer_type == "obb":
                        self.default_infer_type = "detect"
                try:
                    session = RtdetrOnnxSession(
                        p,
                        class_names=self.class_names,
                        imgsz=self.imgsz,
                        device=str(self.device or "cpu"),
                    )
                except Exception as exc:
                    logger.warning("加载 RT-DETRv2 ONNX 失败: %s", exc)
                    return
                self._models[infer_type] = session
                self._names[infer_type] = dict(session.names)
                return
            # YOLO 导出 ONNX：走 Ultralytics
        try:
            from ultralytics import YOLO
        except Exception as exc:
            logger.warning("未安装 ultralytics，无法加载模型: %s", exc)
            return
        task = "obb" if infer_type == "obb" else "detect"
        logger.info("加载%s模型: %s", infer_type, p)
        try:
            model = YOLO(str(p), task=task)
        except Exception as exc:
            logger.warning("加载模型失败 %s: %s", p, exc)
            return
        names = getattr(model, "names", {}) or {}
        if isinstance(names, dict):
            name_map = {int(k): str(v) for k, v in names.items()}
        else:
            name_map = {i: str(n) for i, n in enumerate(names)}
        self._models[infer_type] = model
        self._names[infer_type] = name_map

    def is_ready(self, infer_type: str | None = None) -> bool:
        kind = infer_type or self.default_infer_type
        return self._models.get(kind) is not None

    def readiness_payload(self) -> dict[str, Any]:
        default_type = self.default_infer_type
        detect_loaded = self._models.get("detect") is not None
        obb_loaded = self._models.get("obb") is not None
        ready = self.is_ready(default_type)
        if ready:
            msg = "ok"
        elif default_type == "obb":
            msg = "OBB模型未加载" if not self.obb_model_path else f"模型文件不存在: {self.obb_model_path}"
            if not self.obb_model_path and not self.model_path:
                msg = "未配置 model_path / obb_model_path"
        else:
            msg = "检测模型未加载" if not self.model_path else f"模型文件不存在: {self.model_path}"
            if not self.model_path and not self.obb_model_path:
                msg = "未配置 model_path"
        return {
            "ready": ready,
            "model_loaded": ready,
            "infer_type": default_type,
            "detect_loaded": detect_loaded,
            "obb_loaded": obb_loaded,
            "model_path": self.model_path,
            "obb_model_path": self.obb_model_path,
            "scheme_name": str(self.base_scheme.get("scheme_name") or ""),
            "msg": msg,
        }

    def _detect_bgr(
        self, image_bgr: np.ndarray, *, infer_type: str
    ) -> list[dict[str, Any]]:
        model = self._models.get(infer_type)
        if model is None:
            raise RuntimeError(
                "OBB模型未加载" if infer_type == "obb" else "检测模型未加载"
            )
        names = self._names.get(infer_type) or {}
        h, w = image_bgr.shape[:2]
        raw: list[dict[str, Any]] = []
        with self._infer_lock:
            for x1, y1, x2, y2 in iter_tiles(h, w, self.clip_size, self.overlap_size):
                tile = np.ascontiguousarray(image_bgr[y1:y2, x1:x2])
                if isinstance(model, RtdetrOnnxSession):
                    raw.extend(
                        model.predict_tile(
                            tile, conf=self.conf, ox=x1, oy=y1
                        )
                    )
                    continue
                kwargs: dict[str, Any] = {
                    "conf": self.conf,
                    "iou": self.iou,
                    "verbose": False,
                }
                if self.device:
                    kwargs["device"] = self.device
                preds = model.predict(tile, **kwargs)
                if not preds:
                    continue
                if infer_type == "obb":
                    raw.extend(_parse_obb_pred(preds[0], names, x1, y1))
                else:
                    raw.extend(_parse_detect_pred(preds[0], names, x1, y1))
        return nms_boxes(raw, self.iou)

    def predict_bgr(
        self,
        image_bgr: np.ndarray,
        *,
        image_path: str = "",
        board_id: str = "",
        camera_id: str = "",
        laminating: bool = False,
        mm_per_px: float | None = None,
        scheme_overlay: dict[str, Any] | None = None,
        infer_type: str | None = None,
    ) -> dict[str, Any]:
        kind = normalize_infer_type(infer_type, default=self.default_infer_type)
        scheme = merge_scheme(self.snapshot_scheme(), scheme_overlay)
        scale = float(
            mm_per_px
            if mm_per_px is not None
            else scheme.get("mm_per_px")
            if scheme.get("mm_per_px") is not None
            else self.default_mm_per_px
        )
        dets = self._detect_bgr(image_bgr, infer_type=kind)
        results: list[dict[str, Any]] = []
        for det in dets:
            x1, y1, x2, y2 = det["xyxy"]
            obb = det.get("obb")
            if not isinstance(obb, dict):
                obb = xyxy_to_obb(x1, y1, x2, y2)
            extra: dict[str, Any] = {"infer_type": kind, "obb": obb}
            judged = judge_box(
                name=str(det["name"]),
                score=float(det["score"]),
                left=x1,
                top=y1,
                right=x2,
                bottom=y2,
                scheme=scheme,
                mm_per_px=scale,
                laminating=bool(laminating),
                width_px=float(obb.get("width") or 0),
                height_px=float(obb.get("height") or 0),
                extra=extra,
            )
            if judged is not None:
                results.append(judged)
        counts: dict[str, int] = dict(Counter(str(r["name"]) for r in results))
        ng_count = sum(1 for r in results if r.get("ng"))
        image_ok = ng_count == 0
        h, w = image_bgr.shape[:2]
        return {
            "code": 0,
            "msg": "ok",
            "engine": "scanly-defects-v1",
            "infer_type": kind,
            "board_id": board_id,
            "camera_id": camera_id,
            "scheme_name": str(scheme.get("scheme_name") or ""),
            "laminating": bool(laminating),
            "image_ok": image_ok,
            "board_ok": image_ok,
            "image": {"path": str(image_path or ""), "width": int(w), "height": int(h)},
            "counts": counts,
            "ng_count": ng_count,
            "label_count": sum(1 for r in results if r.get("need_label")),
            "discharge_count": sum(1 for r in results if r.get("need_discharge")),
            "results": results,
        }

    def predict_cameras(
        self,
        cameras: list[dict[str, Any]],
        *,
        board_id: str = "",
        laminating: bool = False,
        mm_per_px: float | None = None,
        scheme_overlay: dict[str, Any] | None = None,
        infer_type: str | None = None,
    ) -> dict[str, Any]:
        kind = normalize_infer_type(infer_type, default=self.default_infer_type)
        grouped: list[dict[str, Any]] = []
        ng_any = False
        for item in cameras:
            if not isinstance(item, dict):
                raise ValueError("请求body必须包含相机与图片路径")
            cam = str(item.get("camera") or "").strip()
            if not cam:
                raise ValueError("camera不能为空")
            images = item.get("images")
            if not isinstance(images, list):
                raise ValueError("images必须是数组")
            slots: list[list[dict[str, Any]]] = []
            for raw_path in images:
                path = str(raw_path or "").strip()
                if not path:
                    raise ValueError("图片路径不能为空")
                one = self.predict_path(
                    path,
                    board_id=board_id,
                    camera_id=cam,
                    laminating=laminating,
                    mm_per_px=mm_per_px,
                    scheme_overlay=scheme_overlay,
                    infer_type=kind,
                )
                slots.append(list(one.get("results") or []))
                if not one.get("image_ok", True):
                    ng_any = True
            grouped.append({"camera": cam, "defects": slots})
        return {
            "code": 0,
            "msg": "ok",
            "engine": "scanly-defects-v1",
            "infer_type": kind,
            "board_id": board_id,
            "board_ok": not ng_any,
            "laminating": bool(laminating),
            "results": grouped,
        }

    def snapshot_scheme(self) -> dict[str, Any]:
        with self._scheme_lock:
            return deepcopy(self.base_scheme)

    def predict_path(self, image_path: str | Path, **kwargs: Any) -> dict[str, Any]:
        img = load_image_bgr(image_path)
        return self.predict_bgr(img, image_path=str(image_path), **kwargs)

    def scheme_view(self) -> dict[str, Any]:
        view = scheme_public_view(self.snapshot_scheme())
        view["code"] = 0
        view["msg"] = "ok"
        return view

    def apply_scheme_update(
        self,
        overlay: dict[str, Any],
        *,
        persist: bool = True,
        replace: bool = False,
    ) -> dict[str, Any]:
        with self._scheme_lock:
            if replace:
                stored = scheme_storage_dict(
                    {
                        "scheme_name": overlay.get("scheme_name")
                        or self.base_scheme.get("scheme_name")
                        or "",
                        "mm_per_px": overlay.get("mm_per_px")
                        if overlay.get("mm_per_px") is not None
                        else self.base_scheme.get("mm_per_px"),
                        "channels": overlay.get("channels")
                        or self.base_scheme.get("channels"),
                        "defects": overlay.get("defects") or [],
                    }
                )
            else:
                stored = scheme_storage_dict(merge_scheme(self.base_scheme, overlay))
            if persist:
                try:
                    save_json(self.scheme_path, stored)
                except OSError as exc:
                    raise RuntimeError("写入方案文件失败") from exc
            self.base_scheme = stored
        view = scheme_public_view(stored)
        view["code"] = 0
        view["msg"] = "ok"
        view["persisted"] = bool(persist)
        return view
