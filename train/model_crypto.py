#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""YOLO ONNX 密文格式：SLYENC01 + nonce + AES-256-GCM(密文||tag)。"""
from __future__ import annotations

import json
import os
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"SLYENC01"
NONCE_LEN = 12
AAD = b"scanly-defects-onnx-v1"
AES_KEY_BYTES = 32


def generate_aes_key() -> bytes:
    return os.urandom(AES_KEY_BYTES)


def encrypt_bytes(plain: bytes, key: bytes) -> bytes:
    if len(key) != AES_KEY_BYTES:
        raise ValueError(f"AES 密钥必须为 {AES_KEY_BYTES} 字节")
    nonce = os.urandom(NONCE_LEN)
    ct = AESGCM(key).encrypt(nonce, plain, AAD)
    return MAGIC + nonce + ct


def decrypt_bytes(blob: bytes, key: bytes) -> bytes:
    if len(key) != AES_KEY_BYTES:
        raise ValueError(f"AES 密钥必须为 {AES_KEY_BYTES} 字节")
    if len(blob) < len(MAGIC) + NONCE_LEN + 16:
        raise ValueError("密文过短")
    if blob[: len(MAGIC)] != MAGIC:
        raise ValueError("不是 scanly ONNX 密文（魔数不匹配）")
    nonce = blob[len(MAGIC) : len(MAGIC) + NONCE_LEN]
    ct = blob[len(MAGIC) + NONCE_LEN :]
    return AESGCM(key).decrypt(nonce, ct, AAD)


def load_key_file(path: str | Path) -> tuple[bytes, dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"密钥文件不是 JSON 对象: {path}")
    import base64

    raw = base64.b64decode(str(data.get("aes_key_b64") or ""))
    if len(raw) != AES_KEY_BYTES:
        raise ValueError(f"密钥文件 aes_key_b64 无效: {path}")
    return raw, data


def save_key_file(path: str | Path, key: bytes, *, model_id: str = "") -> None:
    import base64

    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "model_id": str(model_id or ""),
        "aes_key_b64": base64.b64encode(key).decode("ascii"),
        "note": "厂商侧密钥，禁止随现场安装包分发",
    }
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
