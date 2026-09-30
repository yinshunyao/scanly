#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""将 Ultralytics YOLO .pt 导出为 ONNX 并加密，供 Go 推理服务使用。"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

_TRAIN_DIR = Path(__file__).resolve().parent
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

from model_crypto import (  # noqa: E402
    decrypt_bytes,
    encrypt_bytes,
    generate_aes_key,
    load_key_file,
    save_key_file,
)

logger = logging.getLogger(__name__)

_ENGINE_ERR_HINT = (
    "GET was unable to find an engine：多为 LD_LIBRARY_PATH 中的系统 CUDA11/cuDNN "
    "与 PyTorch wheel 自带库冲突（常见于同一机为 Go ORT 配置了 cuda-11.8）。"
    "请用 SANITIZE_LD_LIBRARY_PATH=True（默认），或 "
    "`LD_LIBRARY_PATH= python3 convert_2_onnx.py`；仍失败可设 HALF=False 导出 FP32。"
)


def resolve_pt_path(raw: str | Path, *, train_dir: Path) -> Path:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("MODEL_PATH 不能为空")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (train_dir / path).resolve()
    return path


def names_to_map(names: Any) -> dict[str, str]:
    if isinstance(names, dict):
        return {str(int(k)): str(v) for k, v in names.items()}
    return {str(i): str(n) for i, n in enumerate(names or [])}


def is_cpu_device(device: str) -> bool:
    d = str(device or "").strip().lower()
    return d == "" or d == "cpu"


def sanitize_ld_library_path(*, enabled: bool) -> None:
    """清空本进程 LD_LIBRARY_PATH，避免系统 cuDNN 覆盖 PyTorch 自带库。

    仅影响当前 Python 进程，不影响调用方 shell。须在首次 import torch/ultralytics 之前调用。
    """
    if not enabled:
        return
    old = os.environ.pop("LD_LIBRARY_PATH", None)
    if old:
        logger.warning(
            "已清空本进程 LD_LIBRARY_PATH（原长度 %d），避免与 PyTorch 自带 CUDA/cuDNN 冲突",
            len(old),
        )


def export_onnx(
    *,
    model_path: Path,
    task: str,
    imgsz: int,
    opset: int,
    nms: bool,
    half: bool,
    device: str,
    sanitize_ld: bool = True,
) -> tuple[Path, dict[str, str]]:
    kind = str(task or "obb").strip().lower()
    if kind not in {"detect", "obb"}:
        raise ValueError(f"不支持的 TASK: {task}")
    if half and is_cpu_device(device):
        raise ValueError(
            "HALF=True 时 Ultralytics 要求 GPU 导出，请将 DEVICE 设为 cuda / 0 等，不能为 cpu"
        )
    # GPU 导出（尤其 half）必须先清路径，再 import ultralytics/torch
    if half or not is_cpu_device(device):
        sanitize_ld_library_path(enabled=sanitize_ld)

    from ultralytics import YOLO

    logger.info(
        "加载权重 task=%s path=%s half=%s device=%s",
        kind,
        model_path,
        half,
        device or "(default)",
    )
    model = YOLO(str(model_path), task=kind)
    try:
        exported = model.export(
            format="onnx",
            imgsz=int(imgsz),
            opset=int(opset),
            simplify=True,
            nms=bool(nms),
            half=bool(half),
            dynamic=False,
            device=device or None,
        )
    except RuntimeError as exc:
        msg = str(exc)
        if "unable to find an engine" in msg.lower() or "GET was unable" in msg:
            raise RuntimeError(f"{msg}\n{_ENGINE_ERR_HINT}") from exc
        raise
    out = Path(str(exported))
    if not out.is_file():
        raise RuntimeError(f"导出未生成 ONNX: {exported}")
    names = names_to_map(getattr(model, "names", {}) or {})
    return out, names


def write_meta(
    path: Path,
    *,
    task: str,
    imgsz: int,
    names: dict[str, str],
    model_id: str,
    half: bool,
) -> None:
    payload = {
        "model_id": str(model_id or ""),
        "task": str(task),
        "imgsz": int(imgsz),
        "names": names,
        "precision": "fp16" if half else "fp32",
        "half": bool(half),
        "encrypted": True,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def convert_and_encrypt(
    *,
    model_path: Path,
    output_dir: Path,
    task: str,
    imgsz: int,
    opset: int,
    nms: bool,
    half: bool,
    device: str,
    model_id: str,
    keep_plain: bool,
    aes_key_path: Path | None,
    verify: bool,
    sanitize_ld: bool = True,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_src, names = export_onnx(
        model_path=model_path,
        task=task,
        imgsz=imgsz,
        opset=opset,
        nms=nms,
        half=half,
        device=device,
        sanitize_ld=sanitize_ld,
    )
    stem = model_id.strip() or onnx_src.stem
    plain_dest = output_dir / f"{stem}.onnx"
    enc_dest = output_dir / f"{stem}.onnx.enc"
    meta_dest = output_dir / f"{stem}.meta.json"
    key_dest = aes_key_path if aes_key_path is not None else output_dir / f"{stem}.key.json"

    if onnx_src.resolve() != plain_dest.resolve():
        shutil.copy2(onnx_src, plain_dest)
        if onnx_src.parent.resolve() == model_path.parent.resolve() and onnx_src.suffix == ".onnx":
            # 训练目录旁的导出文件保留给 Ultralytics；交付物在 OUTPUT_DIR
            pass

    if key_dest.is_file():
        key, _ = load_key_file(key_dest)
        logger.info("复用 AES 密钥: %s", key_dest)
    else:
        key = generate_aes_key()
        save_key_file(key_dest, key, model_id=stem)
        logger.info("已生成 AES 密钥: %s", key_dest)

    plain = plain_dest.read_bytes()
    blob = encrypt_bytes(plain, key)
    enc_dest.write_bytes(blob)
    write_meta(
        meta_dest,
        task=task,
        imgsz=imgsz,
        names=names,
        model_id=stem,
        half=half,
    )
    if verify:
        restored = decrypt_bytes(enc_dest.read_bytes(), key)
        if restored != plain:
            raise RuntimeError("加密往返校验失败")
        logger.info("加密往返校验通过")
    if keep_plain:
        logger.info("保留明文（请人工删除后再交付）: %s", plain_dest)
        if onnx_src.resolve() != plain_dest.resolve() and onnx_src.is_file():
            logger.info("保留 Ultralytics 导出明文（请人工删除）: %s", onnx_src)
    else:
        plain_dest.unlink(missing_ok=True)
        if onnx_src.is_file() and onnx_src.suffix.lower() == ".onnx":
            try:
                onnx_src.unlink()
                logger.info("已删除 Ultralytics 导出明文: %s", onnx_src)
            except OSError:
                logger.warning("无法删除 Ultralytics 导出明文: %s", onnx_src)
        logger.info("已删除交付目录明文 ONNX: %s", plain_dest)
    # 原始 .pt 一律保留，本脚本从不删除
    logger.info("原始 YOLO 权重保留不删: %s", model_path)
    logger.info("精度: %s", "fp16" if half else "fp32")
    logger.info("密文: %s", enc_dest)
    logger.info("元数据: %s", meta_dest)
    return {
        "enc": str(enc_dest),
        "meta": str(meta_dest),
        "key": str(key_dest),
        "onnx": str(plain_dest) if keep_plain else "",
        "precision": "fp16" if half else "fp32",
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    TRAIN_DIR = Path(__file__).resolve().parent

    # —— 输入权重（改完直接运行）——
    MODEL_PATH = "/root/scanly/train/runs/train/weights/best.pt"
    TASK = "obb"  # detect | obb
    MODEL_ID = "v1.3-cam-all-middle"
    IMGSZ = 640
    OPSET = 17
    NMS = False
    # True=FP16（GPU 导出）；False=FP32。半精度须 DEVICE 为 cuda/0，不能 cpu
    HALF = True
    DEVICE = "0"
    # True=GPU 导出前清空本进程 LD_LIBRARY_PATH（避免 ORT/CUDA11 与 torch wheel 冲突）
    SANITIZE_LD_LIBRARY_PATH = True

    # —— 输出与加密 ——
    # 服务器约定：FP16 → predict_go/models/v1.3-onnx-fp16；FP32 → predict_go/models/v1.3-onnx
    _PREDICT_GO_MODELS = Path("/root/scanly/predict_go/models")
    OUTPUT_DIR = _PREDICT_GO_MODELS / ("v1.3-onnx-fp16" if HALF else "v1.3-onnx")
    # True=保留明文 .onnx，由人工删除；False=导出后自动删明文
    KEEP_PLAIN_ONNX = True
    # 空则在 OUTPUT_DIR 下新建 {MODEL_ID}.key.json；已有文件则复用该密钥
    AES_KEY_PATH = ""
    VERIFY_ROUNDTRIP = True

    pt = resolve_pt_path(MODEL_PATH, train_dir=TRAIN_DIR)
    if not pt.is_file():
        raise SystemExit(f"模型文件不存在: {pt}")
    key_path = Path(AES_KEY_PATH).expanduser() if str(AES_KEY_PATH).strip() else None
    if key_path is not None and not key_path.is_absolute():
        key_path = (TRAIN_DIR / key_path).resolve()
    convert_and_encrypt(
        model_path=pt,
        output_dir=Path(OUTPUT_DIR),
        task=TASK,
        imgsz=IMGSZ,
        opset=OPSET,
        nms=NMS,
        half=HALF,
        device=DEVICE,
        model_id=MODEL_ID,
        keep_plain=KEEP_PLAIN_ONNX,
        aes_key_path=key_path,
        verify=VERIFY_ROUNDTRIP,
        sanitize_ld=SANITIZE_LD_LIBRARY_PATH,
    )


if __name__ == "__main__":
    main()
