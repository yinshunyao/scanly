#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FP32 / FP16 ONNX 推理性能对比（Python ORT，复用 predict_go/server_onnx）。

改下方 __main__ 变量后在 IDE 运行。可只配一路模型；两路都配齐时会打印加速比。

依赖：onnxruntime(-gpu)、numpy、Pillow、cryptography（与 predict_go 冒烟一致）。
V100 + ORT 1.18.1：先 source predict_go/env_cuda11.sh 再跑。
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_TEST_DIR = Path(__file__).resolve().parent


def _resolve_predict_go() -> Path:
    """兼容仓库布局 defects/predict_go 与现场扁平布局 scanly/predict_go。"""
    candidates = [
        _TEST_DIR.parent / "predict_go",  # .../defects/predict_go 或 .../scanly/predict_go
        _TEST_DIR.parent.parent / "predict_go",  # test 在 defects/test 时 → scanly/predict_go（若有）
        _TEST_DIR.parent.parent / "defects" / "predict_go",
    ]
    for c in candidates:
        if (c / "server_onnx.py").is_file():
            return c.resolve()
    return (_TEST_DIR.parent / "predict_go").resolve()


def _resolve_train_dir(predict_go: Path) -> Path:
    candidates = [
        predict_go.parent / "train",
        predict_go.parent.parent / "train",
    ]
    for c in candidates:
        if c.is_dir():
            return c.resolve()
    return (predict_go.parent / "train").resolve()


_PREDICT_GO = _resolve_predict_go()
_DEFECTS_DIR = _PREDICT_GO.parent if (_PREDICT_GO.parent / "samples").is_dir() else _TEST_DIR.parent
_TRAIN_DIR = _resolve_train_dir(_PREDICT_GO)

if str(_PREDICT_GO) not in sys.path:
    sys.path.insert(0, str(_PREDICT_GO))
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

import server_onnx as _server_onnx  # noqa: E402
from server_onnx import (  # noqa: E402
    OnnxEngine,
    decrypt_onnx,
    load_aes_key,
    load_rgb,
)

try:
    import inspect

    _ONNX_ENGINE_PARAMS = set(inspect.signature(OnnxEngine.__init__).parameters)
except Exception:  # noqa: BLE001
    _ONNX_ENGINE_PARAMS = set()


@dataclass
class ModelSpec:
    label: str
    model_path: Path
    meta_path: Path | None
    key_path: Path | None  # .key.json 或 license.json；明文 .onnx 可空


@dataclass
class BenchResult:
    label: str
    precision_meta: str
    input_dtype: str
    load_ms: float
    det_count: int
    e2e_ms: list[float]
    ort_ms: list[float]


def resolve_path(raw: str | Path, *, base: Path) -> Path:
    path = Path(str(raw or "").strip()).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def load_meta(path: Path | None, *, task: str) -> dict[str, Any]:
    meta: dict[str, Any] = {"task": task, "imgsz": 640, "names": {}}
    if path is None or not path.is_file():
        return meta
    with path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    if isinstance(loaded, dict):
        meta.update(loaded)
    return meta


def _looks_fp16_onnx(spec: ModelSpec, meta: dict[str, Any]) -> bool:
    p = str(meta.get("precision") or meta.get("dtype") or "").strip().lower()
    if p in {"fp16", "float16", "half", "16"}:
        return True
    blob = f"{spec.label} {spec.model_path}".lower()
    return "fp16" in blob or "float16" in blob


def load_key_bytes(path: Path) -> bytes:
    import base64

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"密钥文件无效: {path}")
    raw = base64.b64decode(str(data.get("aes_key_b64") or ""))
    if len(raw) != 32:
        raise RuntimeError(f"aes_key_b64 无效: {path}")
    return raw


def load_onnx_bytes(model_path: Path, key_path: Path | None) -> bytes:
    blob = model_path.read_bytes()
    suffix = model_path.suffix.lower()
    name = model_path.name.lower()
    if name.endswith(".onnx.enc") or suffix == ".enc":
        if key_path is None or not key_path.is_file():
            raise RuntimeError(f"密文模型需要 KEY_PATH / license: {model_path}")
        # license.json 与 .key.json 均含 aes_key_b64
        if key_path.name == "license.json" or "license" in key_path.name.lower():
            key = load_aes_key(key_path)
        else:
            key = load_key_bytes(key_path)
        return decrypt_onnx(blob, key)
    if suffix == ".onnx":
        return blob
    raise RuntimeError(f"不支持的模型文件: {model_path}（需要 .onnx 或 .onnx.enc）")


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def summarize(ms: list[float]) -> dict[str, float]:
    ordered = sorted(ms)
    return {
        "n": float(len(ms)),
        "mean": float(statistics.fmean(ms)) if ms else float("nan"),
        "p50": percentile(ordered, 50),
        "p95": percentile(ordered, 95),
        "min": float(min(ms)) if ms else float("nan"),
        "max": float(max(ms)) if ms else float("nan"),
        "stdev": float(statistics.stdev(ms)) if len(ms) >= 2 else 0.0,
    }


def sync_cuda() -> None:
    """尽量让 GPU 计时更稳（无 CUDA 时忽略）。"""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass


def bench_model(
    spec: ModelSpec,
    *,
    rgb: np.ndarray,
    device: str,
    task: str,
    conf: float,
    iou: float,
    warmup: int,
    runs: int,
    trt_engine_cache_path: str = "",
    trt_fp16_enable: bool = True,
) -> BenchResult:
    meta = load_meta(spec.meta_path, task=task)
    t0 = time.perf_counter()
    onnx_bytes = load_onnx_bytes(spec.model_path, spec.key_path)
    engine_kwargs: dict[str, Any] = {
        "device": device,
        "conf": conf,
        "iou": iou,
        "task": str(meta.get("task") or task),
    }
    if "trt_engine_cache_path" in _ONNX_ENGINE_PARAMS:
        engine_kwargs["trt_engine_cache_path"] = trt_engine_cache_path
        engine_kwargs["trt_fp16_enable"] = trt_fp16_enable
    elif str(device).strip().lower() in {"tensorrt", "trt", "trt_fp16"}:
        raise RuntimeError(
            "当前导入的 server_onnx.OnnxEngine 不支持 TensorRT 参数。"
            f" 已加载: {getattr(_server_onnx, '__file__', '?')}\n"
            "请把仓库里最新的 predict_go/server_onnx.py 同步到服务器后再跑"
            "（需含 trt_engine_cache_path / build_providers TensorRT 分支）。"
        )
    eng = OnnxEngine(onnx_bytes, meta, **engine_kwargs)
    load_ms = (time.perf_counter() - t0) * 1000.0

    # warmup
    dets: list[Any] = []
    for _ in range(max(0, warmup)):
        sync_cuda()
        dets = eng.predict_rgb(rgb)
        sync_cuda()

    e2e_ms: list[float] = []
    ort_ms: list[float] = []
    from server_onnx import letterbox_chw

    for _ in range(max(1, runs)):
        sync_cuda()
        t1 = time.perf_counter()
        dets = eng.predict_rgb(rgb)
        sync_cuda()
        e2e_ms.append((time.perf_counter() - t1) * 1000.0)

        # 纯 ORT：固定预处理后只跑 session.run
        data, _lb = letterbox_chw(rgb, eng.imgsz)
        feed = data[np.newaxis, ...].astype(eng.input_dtype, copy=False)
        sync_cuda()
        t2 = time.perf_counter()
        _ = eng.session.run([eng.output_name], {eng.input_name: feed})
        sync_cuda()
        ort_ms.append((time.perf_counter() - t2) * 1000.0)

    precision_meta = str(meta.get("precision") or ("fp16" if meta.get("half") else "") or "unknown")
    return BenchResult(
        label=spec.label,
        precision_meta=precision_meta,
        input_dtype=str(np.dtype(eng.input_dtype)),
        load_ms=load_ms,
        det_count=len(dets),
        e2e_ms=e2e_ms,
        ort_ms=ort_ms,
    )


def print_result(r: BenchResult) -> None:
    e2e = summarize(r.e2e_ms)
    ort = summarize(r.ort_ms)
    print(f"\n=== {r.label} ===")
    print(
        f"meta.precision={r.precision_meta}  session.input_dtype={r.input_dtype}  "
        f"load_ms={r.load_ms:.1f}  last_dets={r.det_count}"
    )
    print(
        "e2e(letterbox+run+decode+nms) ms: "
        f"mean={e2e['mean']:.2f}  p50={e2e['p50']:.2f}  p95={e2e['p95']:.2f}  "
        f"min={e2e['min']:.2f}  max={e2e['max']:.2f}  stdev={e2e['stdev']:.2f}  n={int(e2e['n'])}"
    )
    print(
        "ort(session.run only) ms:        "
        f"mean={ort['mean']:.2f}  p50={ort['p50']:.2f}  p95={ort['p95']:.2f}  "
        f"min={ort['min']:.2f}  max={ort['max']:.2f}  stdev={ort['stdev']:.2f}  n={int(ort['n'])}"
    )


def print_speedup(base: BenchResult, other: BenchResult) -> None:
    b = summarize(base.e2e_ms)["mean"]
    o = summarize(other.e2e_ms)["mean"]
    rb = summarize(base.ort_ms)["mean"]
    ro = summarize(other.ort_ms)["mean"]
    if b > 0 and o > 0:
        print(
            f"\n加速比（相对 {base.label}，e2e mean）: {base.label}/{other.label} = {b / o:.3f}x"
        )
    if rb > 0 and ro > 0:
        print(
            f"加速比（相对 {base.label}，ort mean）: {base.label}/{other.label} = {rb / ro:.3f}x"
        )


def main() -> None:
    PREDICT_GO = _PREDICT_GO

    # —— 设备与计次 ——
    # cpu | cuda | tensorrt（别名 trt / trt_fp16）
    # DEVICE = "cuda"
    DEVICE = "tensorrt"
    TRT_FP16_ENABLE = True
    # TensorRT 请用 FP32 ONNX + EP 内 FP16；FP16 ONNX 再叠 TRT FP16 易 NaN
    TRT_SKIP_FP16_ONNX = True
    TRT_ENGINE_CACHE_PATH = str(PREDICT_GO / "models" / "trt_cache")
    TASK = "obb"
    CONF = 0.25
    IOU = 0.45
    WARMUP = 5
    RUNS = 30

    # —— 测试图 ——
    sample_roots = [
        _DEFECTS_DIR / "samples",
        PREDICT_GO.parent / "samples",
        _TEST_DIR.parent / "samples",
    ]
    default_sample = None
    for root in sample_roots:
        if not root.is_dir():
            continue
        default_sample = next(
            (
                p
                for p in sorted(root.iterdir())
                if p.is_file() and p.suffix.lower() in {".bmp", ".jpg", ".jpeg"}
            ),
            None,
        )
        if default_sample is not None:
            break
    IMAGE_PATH = str(default_sample) if default_sample else ""
    # —— 模型（缺文件的条目会跳过并提示）——
    # 服务器目录约定（相对 predict_go/）：
    #   全精度 FP32 → models/v1.3-onnx/
    #   半精度 FP16 → models/v1.3-onnx-fp16/
    # key：.key.json 或 predict_go/config/license.json；明文 .onnx 可把 KEY 留空
    FP32 = {
        "label": "fp32",
        "model": "models/v1.3-onnx/v1.3-cam-all-middle.onnx.enc",
        "meta": "models/v1.3-onnx/v1.3-cam-all-middle.meta.json",
        "key": "config/license.json",
    }
    FP16 = {
        "label": "fp16",
        "model": "models/v1.3-onnx-fp16/v1.3-cam-all-middle.onnx.enc",
        "meta": "models/v1.3-onnx-fp16/v1.3-cam-all-middle.meta.json",
        "key": "config/license.json",
    }
    # 只测一路时把另一路 MODEL 路径留空字符串即可
    MODELS = [FP32, FP16]

    image = resolve_path(IMAGE_PATH, base=_DEFECTS_DIR)
    if not image.is_file():
        # 现场常见 samples 在 scanly/samples
        image = resolve_path(IMAGE_PATH, base=_PREDICT_GO.parent)
    if not image.is_file():
        raise SystemExit(f"测试图不存在: {IMAGE_PATH}")
    rgb = load_rgb(str(image))
    print(f"[import] server_onnx={getattr(_server_onnx, '__file__', '?')}")
    print(
        f"image={image} shape={rgb.shape} device={DEVICE} "
        f"trt_fp16={TRT_FP16_ENABLE} warmup={WARMUP} runs={RUNS}"
    )
    if "trt_engine_cache_path" not in _ONNX_ENGINE_PARAMS:
        print(
            "[warn] OnnxEngine 无 TRT 参数：仍可测 cuda/cpu；"
            "测 tensorrt 前请同步最新 predict_go/server_onnx.py"
        )

    results: list[BenchResult] = []
    for raw in MODELS:
        label = str(raw.get("label") or "model")
        model_raw = str(raw.get("model") or "").strip()
        if not model_raw:
            print(f"[skip] {label}: model 为空")
            continue
        model_path = resolve_path(model_raw, base=PREDICT_GO)
        if not model_path.is_file():
            print(f"[skip] {label}: 模型文件不存在 {model_path}")
            continue
        meta_raw = str(raw.get("meta") or "").strip()
        key_raw = str(raw.get("key") or "").strip()
        meta_path = resolve_path(meta_raw, base=PREDICT_GO) if meta_raw else None
        key_path = resolve_path(key_raw, base=PREDICT_GO) if key_raw else None
        spec = ModelSpec(
            label=label,
            model_path=model_path,
            meta_path=meta_path,
            key_path=key_path,
        )
        meta_preview = load_meta(spec.meta_path, task=TASK)
        if (
            str(DEVICE).strip().lower() in {"tensorrt", "trt", "trt_fp16"}
            and TRT_SKIP_FP16_ONNX
            and _looks_fp16_onnx(spec, meta_preview)
        ):
            print(
                f"[skip] {label}: TensorRT 默认不测 FP16 ONNX "
                "（用 FP32 密文 + trt_fp16；叠半精度易 NaN）。"
                "若要强制测，把 TRT_SKIP_FP16_ONNX = False"
            )
            continue
        print(f"\n[load] {label} <- {model_path}")
        result = bench_model(
            spec,
            rgb=rgb,
            device=DEVICE,
            task=TASK,
            conf=CONF,
            iou=IOU,
            warmup=WARMUP,
            runs=RUNS,
            trt_engine_cache_path=TRT_ENGINE_CACHE_PATH,
            trt_fp16_enable=TRT_FP16_ENABLE,
        )
        print_result(result)
        results.append(result)

    if not results:
        raise SystemExit("没有可测模型：请检查 MODELS 路径（需至少一路 FP32 或 FP16）")

    by_label = {r.label.lower(): r for r in results}
    if "fp32" in by_label and "fp16" in by_label:
        print_speedup(by_label["fp32"], by_label["fp16"])
    elif len(results) == 2:
        print_speedup(results[0], results[1])


if __name__ == "__main__":
    main()
