#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Python 冒烟推理服务：复用 Go 的 predict.json / License / 密文 ONNX。

用途：在现场验证「同一套模型 + GPU」用 pip 的 onnxruntime(-gpu) 能否跑通。
与 Go 的区别：Python 使用 pip 安装的 ORT，不加载 third_party 下载的 C 库。

接口（与 Go 子集对齐，便于 test_predict_path_api.py）：
  GET  /health/ready
  GET  /
  POST /v1/predict
  POST /scanly_predict

判定方案（scheme）仅做简化：检出框原样返回，ng=True（冒烟用，非产线判定）。
"""

from __future__ import annotations

import base64
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent

MAGIC = b"SLYENC01"
AAD = b"scanly-defects-onnx-v1"
NONCE_LEN = 12


def _resolve(root: Path, p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def load_predict_json(root: Path) -> dict[str, Any]:
    cfg_path = root / "config" / "predict.json"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise RuntimeError(f"无效配置: {cfg_path}")
    return cfg


def decrypt_onnx(blob: bytes, key: bytes) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if len(key) != 32:
        raise RuntimeError("AES 密钥长度无效（需要 32 字节）")
    if len(blob) < len(MAGIC) + NONCE_LEN + 16:
        raise RuntimeError("密文过短")
    if blob[: len(MAGIC)] != MAGIC:
        raise RuntimeError("不是 scanly ONNX 密文（魔数不匹配）")
    nonce = blob[len(MAGIC) : len(MAGIC) + NONCE_LEN]
    ct = blob[len(MAGIC) + NONCE_LEN :]
    try:
        return AESGCM(key).decrypt(nonce, ct, AAD)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("模型解密失败") from exc


def load_aes_key(license_path: Path) -> bytes:
    with license_path.open("r", encoding="utf-8") as f:
        lic = json.load(f)
    raw = base64.b64decode(str(lic.get("aes_key_b64") or ""))
    if len(raw) != 32:
        raise RuntimeError(f"license aes_key_b64 无效: {license_path}")
    return raw


def want_cuda(device: str) -> bool:
    d = (device or "").strip().lower()
    return d in ("cuda", "gpu") or d.startswith("cuda:")


def want_tensorrt(device: str) -> bool:
    d = (device or "").strip().lower()
    return d in ("tensorrt", "trt", "trt_fp16")


def want_gpu(device: str) -> bool:
    return want_cuda(device) or want_tensorrt(device)


def onnx_is_fp16(meta: dict[str, Any]) -> bool:
    p = str(meta.get("precision") or meta.get("dtype") or "").strip().lower()
    if p in {"fp16", "float16", "half", "16"}:
        return True
    half = meta.get("half")
    return half is True or str(half).strip().lower() in {"1", "true", "yes"}


def build_providers(
    device: str,
    *,
    trt_engine_cache_path: str = "",
    trt_fp16_enable: bool = True,
) -> list[str | tuple[str, dict[str, Any]]]:
    import onnxruntime as ort

    avail = ort.get_available_providers()
    print(f"[ort] package={getattr(ort, '__file__', '?')}")
    print(f"[ort] version={ort.__version__} available={avail}")

    cuda_opts = {
        "device_id": 0,
        "cudnn_conv_algo_search": "DEFAULT",
        "cudnn_conv_use_max_workspace": "1",
    }

    if want_tensorrt(device):
        if "TensorrtExecutionProvider" not in avail:
            raise RuntimeError(
                "配置要求 TensorRT，但当前 Python ORT 无 TensorrtExecutionProvider。"
                "请安装带 TensorRT 的 onnxruntime-gpu，并确保系统 TensorRT 库可用。"
            )
        if "CUDAExecutionProvider" not in avail:
            raise RuntimeError(
                "配置要求 TensorRT，但缺少 CUDAExecutionProvider（TRT 未覆盖算子需 CUDA 回退）。"
            )
        cache = str(trt_engine_cache_path or "").strip() or "models/trt_cache"
        Path(cache).mkdir(parents=True, exist_ok=True)
        trt_opts: dict[str, Any] = {
            "device_id": 0,
            "trt_fp16_enable": bool(trt_fp16_enable),
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": cache,
        }
        print(
            f"[ort] TensorRT EP fp16={bool(trt_fp16_enable)} cache={cache}"
        )
        return [
            ("TensorrtExecutionProvider", trt_opts),
            ("CUDAExecutionProvider", cuda_opts),
            "CPUExecutionProvider",
        ]

    if want_cuda(device):
        if "CUDAExecutionProvider" not in avail:
            raise RuntimeError(
                "配置要求 GPU，但当前 Python ORT 无 CUDAExecutionProvider。"
                "请安装 onnxruntime-gpu（卸掉纯 CPU 的 onnxruntime）。"
            )
        return [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


@dataclass
class LetterboxMeta:
    ratio: float
    pad_x: float
    pad_y: float


@dataclass
class Det:
    name: str
    score: float
    xyxy: tuple[float, float, float, float]
    obb: dict[str, Any]


class OnnxEngine:
    def __init__(
        self,
        onnx_bytes: bytes,
        meta: dict[str, Any],
        *,
        device: str,
        conf: float,
        iou: float,
        task: str,
        trt_engine_cache_path: str = "",
        trt_fp16_enable: bool = True,
    ) -> None:
        import onnxruntime as ort

        self.conf = conf if conf > 0 else 0.25
        self.iou = iou if iou > 0 else 0.45
        self.imgsz = int(meta.get("imgsz") or 640) or 640
        self.task = (meta.get("task") or task or "obb").strip().lower()
        names_raw = meta.get("names") or {}
        self.names: dict[int, str] = {}
        if isinstance(names_raw, dict):
            for k, v in names_raw.items():
                try:
                    self.names[int(k)] = str(v)
                except (TypeError, ValueError):
                    continue
        if want_tensorrt(device) and onnx_is_fp16(meta) and trt_fp16_enable:
            print(
                "[ort] ONNX 已是 FP16，关闭额外 TensorRT FP16"
                "（叠半精度在 V100/TRT8.6 上易出 NaN）"
            )
            trt_fp16_enable = False
        providers = build_providers(
            device,
            trt_engine_cache_path=trt_engine_cache_path,
            trt_fp16_enable=trt_fp16_enable,
        )
        so = ort.SessionOptions()
        so.log_severity_level = 2
        self.session = ort.InferenceSession(onnx_bytes, sess_options=so, providers=providers)
        used = self.session.get_providers()
        print(f"[ort] session providers={used}")
        if want_tensorrt(device):
            if not used or "Tensorrt" not in str(used[0]):
                raise RuntimeError(
                    "要求 TensorRT，但会话实际 providers="
                    f"{used}。常见原因：系统缺少与 ORT 匹配的 TensorRT 动态库"
                    "（本机日志常见 libnvinfer.so.10: No such file or directory）。\n"
                    "处理：安装 TensorRT 10.x（或 ORT 文档要求的版本），并将含"
                    " libnvinfer.so.* 的目录加入 LD_LIBRARY_PATH 后重试；"
                    "也可用 `ldconfig -p | grep nvinfer` / `find /usr -name 'libnvinfer.so*'`"
                    " 排查。临时可改回 device=cuda 仅用 CUDA EP。"
                )
        elif want_cuda(device) and (
            not used or not str(used[0]).startswith("CUDA")
        ):
            raise RuntimeError(f"要求 CUDA，但会话实际 providers={used}")
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        in_type = str(self.session.get_inputs()[0].type or "")
        self.input_dtype = np.float16 if "float16" in in_type else np.float32
        print(f"[ort] input type={in_type} feed_dtype={self.input_dtype}")

    def predict_rgb(self, rgb: np.ndarray, ox: int = 0, oy: int = 0) -> list[Det]:
        data, lb = letterbox_chw(rgb, self.imgsz)
        feed = data[np.newaxis, ...].astype(self.input_dtype, copy=False)
        outs = self.session.run(
            [self.output_name],
            {self.input_name: feed},
        )
        out = np.asarray(outs[0], dtype=np.float32)
        return nms(decode_yolo(out, self.task, self.names, self.conf, lb, ox, oy), self.iou)


def letterbox_chw(rgb: np.ndarray, size: int) -> tuple[np.ndarray, LetterboxMeta]:
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    r = min(size / max(w, 1), size / max(h, 1))
    new_w = max(1, int(round(w * r)))
    new_h = max(1, int(round(h * r)))
    from PIL import Image

    resized = np.asarray(
        Image.fromarray(rgb).resize((new_w, new_h), Image.BILINEAR),
        dtype=np.uint8,
    )
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - new_w) / 2.0
    pad_y = (size - new_h) / 2.0
    x0 = int(round(pad_x))
    y0 = int(round(pad_y))
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    chw = canvas.astype(np.float32).transpose(2, 0, 1) / 255.0
    return chw, LetterboxMeta(ratio=r, pad_x=pad_x, pad_y=pad_y)


def axis_starts(length: int, clip: int, overlap: int) -> list[int]:
    if clip >= length:
        return [0]
    step = max(1, clip - max(0, overlap))
    starts = list(range(0, length - clip + 1, step))
    if starts[-1] != length - clip:
        starts.append(length - clip)
    return starts


def tiles(h: int, w: int, clip_size: int, overlap: int) -> list[tuple[int, int, int, int]]:
    if clip_size <= 0:
        return [(0, 0, w, h)]
    clip_w = min(clip_size, w)
    clip_h = min(clip_size, h)
    out: list[tuple[int, int, int, int]] = []
    for y0 in axis_starts(h, clip_h, overlap):
        for x0 in axis_starts(w, clip_w, overlap):
            out.append((x0, y0, x0 + clip_w, y0 + clip_h))
    return out


def xywhr_to_points(cx: float, cy: float, w: float, h: float, angle: float) -> list[list[int]] | None:
    vals = (cx, cy, w, h, angle)
    if not all(math.isfinite(float(v)) for v in vals):
        return None
    cos_v = math.cos(angle)
    sin_v = math.sin(angle)
    vx1 = (w / 2) * cos_v
    vy1 = (w / 2) * sin_v
    vx2 = -(h / 2) * sin_v
    vy2 = (h / 2) * cos_v
    pts = [
        (cx + vx1 + vx2, cy + vy1 + vy2),
        (cx + vx1 - vx2, cy + vy1 - vy2),
        (cx - vx1 - vx2, cy - vy1 - vy2),
        (cx - vx1 + vx2, cy - vy1 + vy2),
    ]
    if not all(math.isfinite(x) and math.isfinite(y) for x, y in pts):
        return None
    return [[int(round(x)), int(round(y))] for x, y in pts]


def xyxy_to_obb(x1: float, y1: float, x2: float, y2: float) -> dict[str, Any]:
    left, right = min(x1, x2), max(x1, x2)
    top, bottom = min(y1, y2), max(y1, y2)
    return {
        "points": [
            [int(round(left)), int(round(top))],
            [int(round(right)), int(round(top))],
            [int(round(right)), int(round(bottom))],
            [int(round(left)), int(round(bottom))],
        ],
        "cx": round((left + right) / 2, 2),
        "cy": round((top + bottom) / 2, 2),
        "width": round(max(right - left, 0), 2),
        "height": round(max(bottom - top, 0), 2),
        "angle": 0.0,
    }


def decode_yolo(
    out: np.ndarray,
    task: str,
    names: dict[int, str],
    conf: float,
    lb: LetterboxMeta,
    ox: int,
    oy: int,
) -> list[Det]:
    arr = np.asarray(out, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        return []
    # (C, N) or (N, C)
    if arr.shape[0] < arr.shape[1] and arr.shape[0] < 128:
        channel_first = True
        channels, anchors = int(arr.shape[0]), int(arr.shape[1])
        at = lambda ch, anc: float(arr[ch, anc])  # noqa: E731
    else:
        channel_first = False
        anchors, channels = int(arr.shape[0]), int(arr.shape[1])
        at = lambda ch, anc: float(arr[anc, ch])  # noqa: E731
    _ = channel_first
    is_obb = task == "obb"
    min_c = 6 if is_obb else 5
    if channels < min_c or anchors <= 0:
        return []
    nc = channels - 5 if is_obb else channels - 4
    if nc < 1:
        nc = 1
    dets: list[Det] = []
    for i in range(anchors):
        best = 0
        best_score = at(4, i)
        if nc > 1:
            best_score = 0.0
            for c in range(nc):
                s = at(4 + c, i)
                if s > best_score:
                    best_score = s
                    best = c
        if best_score < conf:
            continue
        cx = (at(0, i) - lb.pad_x) / lb.ratio + ox
        cy = (at(1, i) - lb.pad_y) / lb.ratio + oy
        bw = at(2, i) / lb.ratio
        bh = at(3, i) / lb.ratio
        if not all(math.isfinite(float(v)) for v in (cx, cy, bw, bh, best_score)):
            continue
        name = names.get(best, str(best))
        if is_obb:
            angle = at(channels - 1, i)
            pts = xywhr_to_points(cx, cy, bw, bh, angle)
            if pts is None:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            xyxy = (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))
            obb = {
                "points": pts,
                "cx": round(cx, 2),
                "cy": round(cy, 2),
                "width": round(bw, 2),
                "height": round(bh, 2),
                "angle": round(angle, 6),
            }
        else:
            x1, y1 = cx - bw / 2, cy - bh / 2
            x2, y2 = cx + bw / 2, cy + bh / 2
            xyxy = (x1, y1, x2, y2)
            obb = xyxy_to_obb(x1, y1, x2, y2)
        dets.append(Det(name=name, score=float(best_score), xyxy=xyxy, obb=obb))
        if len(dets) >= 3000:
            break
    return dets


def iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(dets: list[Det], iou_thresh: float) -> list[Det]:
    order = sorted(range(len(dets)), key=lambda i: dets[i].score, reverse=True)
    keep: list[Det] = []
    while order:
        i = order[0]
        keep.append(dets[i])
        remain: list[int] = []
        for j in order[1:]:
            if dets[i].name == dets[j].name and iou_xyxy(dets[i].xyxy, dets[j].xyxy) >= iou_thresh:
                continue
            remain.append(j)
        order = remain
    return keep


def load_rgb(path: str) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


class App:
    def __init__(self, root: Path, cfg: dict[str, Any], engines: dict[str, OnnxEngine]) -> None:
        self.root = root
        self.cfg = cfg
        self.engines = engines
        self.default_type = str(cfg.get("infer_type") or "obb").strip().lower() or "obb"

    def readiness(self) -> dict[str, Any]:
        ready = self.default_type in self.engines
        return {
            "ready": ready,
            "model_loaded": ready,
            "infer_type": self.default_type,
            "detect_loaded": "detect" in self.engines,
            "obb_loaded": "obb" in self.engines,
            "model_path": self.cfg.get("model_enc_path") or "",
            "obb_model_path": self.cfg.get("obb_model_enc_path") or "",
            "license_ok": True,
            "engine": "scanly-defects-py-onnx-smoke",
            "msg": "ok" if ready else "模型未加载",
        }

    def predict(self, body: Any) -> dict[str, Any]:
        extras: dict[str, Any]
        cameras: Any
        if isinstance(body, list):
            extras, cameras = {}, body
        elif isinstance(body, dict):
            extras = body
            cameras = body.get("cameras") or body.get("items")
            if cameras is None and body.get("camera") is not None:
                cameras = [{"camera": body.get("camera"), "images": body.get("images")}]
        else:
            raise ValueError("请求body必须是JSON")
        if not isinstance(cameras, list) or not cameras:
            raise ValueError("请求body必须包含相机与图片路径")

        kind = str(extras.get("infer_type") or "").strip().lower()
        if not kind:
            kind = self.default_type
        if kind not in ("obb", "detect"):
            raise ValueError("不支持的推理类型")
        eng = self.engines.get(kind)
        if eng is None:
            raise ValueError("OBB模型未加载" if kind == "obb" else "检测模型未加载")

        clip_size = int(self.cfg.get("clip_size") or 0)
        overlap = int(self.cfg.get("overlap_size") or 0)
        t0 = time.time()
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
                rgb = load_rgb(path)
                h, w = rgb.shape[:2]
                raw: list[Det] = []
                for x1, y1, x2, y2 in tiles(h, w, clip_size, overlap):
                    tile = rgb[y1:y2, x1:x2]
                    raw.extend(eng.predict_rgb(tile, ox=x1, oy=y1))
                raw = nms(raw, eng.iou)
                items = [
                    {
                        "name": d.name,
                        "score": round(d.score, 4),
                        "ng": True,
                        "xyxy": [round(v, 2) for v in d.xyxy],
                        "obb": d.obb,
                        "infer_type": kind,
                    }
                    for d in raw
                ]
                slots.append(items)
                if items:
                    ng_any = True
            grouped.append({"camera": cam, "defects": slots})
        return {
            "code": 0,
            "msg": "ok",
            "engine": "scanly-defects-py-onnx-smoke",
            "infer_type": kind,
            "board_id": str(extras.get("board_id") or ""),
            "board_ok": not ng_any,
            "laminating": bool(extras.get("laminating")),
            "elapsed_ms": round((time.time() - t0) * 1000.0, 2),
            "results": grouped,
        }


def create_app(root: Path | None = None) -> App:
    root = root or ROOT
    cfg = load_predict_json(root)
    lic_path = _resolve(root, str(cfg.get("license_path") or "config/license.json"))
    key = load_aes_key(lic_path)
    device = str(cfg.get("device") or "cpu")
    conf = float(cfg.get("conf") or 0.25)
    iou = float(cfg.get("iou") or 0.45)
    infer_type = str(cfg.get("infer_type") or "obb").strip().lower() or "obb"
    trt_fp16_enable = bool(cfg["trt_fp16_enable"]) if "trt_fp16_enable" in cfg else True
    trt_cache_raw = str(cfg.get("trt_engine_cache_path") or "models/trt_cache").strip()
    trt_cache = str(_resolve(root, trt_cache_raw)) if trt_cache_raw else ""

    detect_enc = str(cfg.get("model_enc_path") or "").strip()
    obb_enc = str(cfg.get("obb_model_enc_path") or "").strip()
    if infer_type == "obb" and not obb_enc and detect_enc:
        obb_enc, detect_enc = detect_enc, ""

    engines: dict[str, OnnxEngine] = {}

    def _load(enc_rel: str, meta_rel: str, task: str) -> OnnxEngine:
        enc_path = _resolve(root, enc_rel)
        meta_path = _resolve(root, meta_rel) if meta_rel else None
        meta: dict[str, Any] = {"task": task, "imgsz": 640, "names": {}}
        if meta_path and meta_path.is_file():
            with meta_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                meta.update(loaded)
        print(f"[model] decrypt {enc_path}")
        plain = decrypt_onnx(enc_path.read_bytes(), key)
        print(f"[model] onnx bytes={len(plain)} task={task} device={device}")
        return OnnxEngine(
            plain,
            meta,
            device=device,
            conf=conf,
            iou=iou,
            task=task,
            trt_engine_cache_path=trt_cache,
            trt_fp16_enable=trt_fp16_enable,
        )
    if detect_enc:
        engines["detect"] = _load(detect_enc, str(cfg.get("detect_meta_path") or ""), "detect")
    if obb_enc:
        engines["obb"] = _load(obb_enc, str(cfg.get("obb_meta_path") or ""), "obb")
    if not engines:
        raise RuntimeError("predict.json 未配置 model_enc_path / obb_model_enc_path")
    return App(root, cfg, engines)


def make_handler(app: App) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def _write_json(self, status: int, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _read_json(self) -> Any:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length > 0 else b"{}"
            return json.loads(raw.decode("utf-8"))

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path in ("/", ""):
                self._write_json(
                    200,
                    {
                        "name": "scanly-defects-py-onnx-smoke",
                        "engine": "scanly-defects-py-onnx-smoke",
                        "note": "pip onnxruntime；配置与 Go 共用 config/predict.json",
                        "endpoints": ["GET /health/ready", "POST /v1/predict"],
                    },
                )
                return
            if path == "/health/ready":
                payload = app.readiness()
                self._write_json(200 if payload.get("ready") else 503, payload)
                return
            self._write_json(404, {"code": 1, "msg": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path not in ("/v1/predict", "/scanly_predict"):
                self._write_json(404, {"code": 1, "msg": "not found"})
                return
            try:
                body = self._read_json()
                payload = app.predict(body)
                self._write_json(200, payload)
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                self._write_json(200, {"code": 1, "msg": str(exc)})

    return Handler


def main(
    *,
    root: Path | None = None,
    host: str | None = None,
    port: int | None = None,
) -> None:
    root = root or ROOT
    app = create_app(root)
    cfg = app.cfg
    bind_host = host if host is not None else str(cfg.get("host") or "0.0.0.0")
    bind_port = int(port if port is not None else (cfg.get("port") or 37871))
    ready = app.readiness()
    print(f"[ready] {ready}")
    print(
        f"[listen] http://{bind_host}:{bind_port}  "
        f"(Python 冒烟；device={cfg.get('device')})"
    )
    print("[note] 与 Go 冲突时请先停 Go，或把下方 PORT 改成 37872")
    httpd = ThreadingHTTPServer((bind_host, bind_port), make_handler(app))
    httpd.serve_forever()


if __name__ == "__main__":
    # 与 Go 同机测试时建议改 PORT，避免抢 37871
    HOST: str | None = None  # None → predict.json host
    PORT: int | None = 37872  # None → predict.json port；默认错开 Go
    main(host=HOST, port=PORT)
