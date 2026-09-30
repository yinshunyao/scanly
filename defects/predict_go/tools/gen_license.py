#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""签发 GPU License：私钥签名；每张卡绑定 uuid/name/serial 三项。"""
from __future__ import annotations

import base64
import csv
import io
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

_TOOLS_DIR = Path(__file__).resolve().parent
_GO_ROOT = _TOOLS_DIR.parent
_TRAIN_DIR = _GO_ROOT.parents[1] / "train"
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

from model_crypto import load_key_file  # noqa: E402

logger = logging.getLogger(__name__)

CANONICAL_SEP = (",", ":")
SMI_QUERY = ["--query-gpu=uuid,name,serial", "--format=csv,noheader"]


def _nvidia_smi_candidates() -> list[str]:
    out = ["nvidia-smi"]
    if sys.platform == "win32":
        out.extend(
            [
                r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
                r"C:\Windows\System32\nvidia-smi.exe",
            ]
        )
    return out


def parse_gpu_csv(raw: str) -> list[dict[str, str]]:
    gpus: list[dict[str, str]] = []
    reader = csv.reader(io.StringIO(raw))
    for row in reader:
        if len(row) < 3:
            continue
        item = {
            "uuid": str(row[0]).strip(),
            "name": str(row[1]).strip(),
            "serial": str(row[2]).strip(),
        }
        if item["uuid"] and item["name"] and item["serial"]:
            gpus.append(item)
    return gpus


def list_gpus() -> list[dict[str, str]]:
    last_err = ""
    for exe in _nvidia_smi_candidates():
        try:
            raw = subprocess.check_output(
                [exe, *SMI_QUERY],
                stderr=subprocess.STDOUT,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            last_err = str(exc)
            continue
        found = parse_gpu_csv(raw)
        if found:
            return found
    if last_err:
        logger.warning("读取 GPU 信息失败: %s", last_err)
    return []


def normalize_gpu(item: Any) -> dict[str, str] | None:
    if not isinstance(item, dict):
        return None
    gpu = {
        "uuid": str(item.get("uuid") or "").strip(),
        "name": str(item.get("name") or "").strip(),
        "serial": str(item.get("serial") or "").strip(),
    }
    if not gpu["uuid"] or not gpu["name"] or not gpu["serial"]:
        return None
    return gpu


def generate_keypair(private_path: Path, public_path: Path) -> None:
    key = Ed25519PrivateKey.generate()
    private_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    pub_pem = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    public_path.write_bytes(pub_pem)
    embed = _GO_ROOT / "internal" / "license" / "license_ed25519.pub"
    embed.parent.mkdir(parents=True, exist_ok=True)
    embed.write_bytes(pub_pem)
    logger.info("已生成私钥 %s （勿分发）", private_path)
    logger.info("已生成公钥 %s 并写入 Go embed %s", public_path, embed)
    logger.info("更换密钥后必须重新编译 Go 服务")


def load_private_key(path: Path) -> Ed25519PrivateKey:
    data = path.read_bytes()
    key = serialization.load_pem_private_key(data, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError(f"不是 Ed25519 私钥: {path}")
    return key


def canonical_payload(data: dict) -> bytes:
    body = {k: v for k, v in data.items() if k != "signature_b64"}
    text = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=CANONICAL_SEP)
    return text.encode("utf-8")


def sign_license(data: dict, private_key: Ed25519PrivateKey) -> dict:
    payload = dict(data)
    payload.pop("signature_b64", None)
    sig = private_key.sign(canonical_payload(payload))
    payload["signature_b64"] = base64.b64encode(sig).decode("ascii")
    return payload


def verify_license(data: dict, public_pem: bytes) -> None:
    pub = serialization.load_pem_public_key(public_pem)
    if not isinstance(pub, Ed25519PublicKey):
        raise ValueError("公钥不是 Ed25519")
    sig_b64 = str(data.get("signature_b64") or "")
    if not sig_b64:
        raise ValueError("缺少 signature_b64")
    pub.verify(base64.b64decode(sig_b64), canonical_payload(data))


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    KEYS_DIR = _GO_ROOT / "keys"
    PRIVATE_KEY_PATH = KEYS_DIR / "license_ed25519.pem"
    PUBLIC_KEY_PATH = KEYS_DIR / "license_ed25519.pub"

    # True：只生成/覆盖密钥对后退出（会同步写入 Go embed 公钥）
    INIT_KEYS = False
    # True：打印本机 GPU uuid/name/serial 后退出（现场登记用）
    LIST_LOCAL_GPUS = False

    # todo 需要修改
    AES_KEY_PATH = _TRAIN_DIR / "output" / "onnx" / "v1.3-cam-all-middle.key.json"
    # nvidia-smi --query-gpu=uuid,name,serial --format=csv,noheader
    GPUS: list[dict[str, str]] = [
        {
            "uuid": "GPU-aec2d77c-763b-4295-4875-55b9887cd265",
            "name": "Tesla V100S-PCIE-32GB",
            "serial": "1562721015128",
        },
    ]
    # True 且 GPUS 为空时，把本机 nvidia-smi 三列写入 License
    USE_LOCAL_GPUS_IF_EMPTY = False
    CUSTOMER = ""
    MODEL_ID = "cam123-0820-obb"
    NOTE = ""
    EXPIRES_AT = ""  # 空=不过期；例 "2027-12-31T23:59:59+08:00"
    OUTPUT_PATH = _GO_ROOT / "config" / "license.json"
    VERIFY_AFTER_SIGN = True

    if INIT_KEYS:
        generate_keypair(PRIVATE_KEY_PATH, PUBLIC_KEY_PATH)
        return

    if LIST_LOCAL_GPUS:
        found = list_gpus()
        if not found:
            raise SystemExit("未检测到 NVIDIA GPU，请确认已安装驱动且 nvidia-smi 可用")
        for item in found:
            print(f"{item['uuid']}, {item['name']}, {item['serial']}")
        print(json.dumps(found, ensure_ascii=False, indent=2))
        return

    gpus = [g for item in GPUS if (g := normalize_gpu(item))]
    if not gpus and USE_LOCAL_GPUS_IF_EMPTY:
        gpus = list_gpus()
        logger.info("使用本机 GPU: %s", gpus)
    if not gpus:
        raise SystemExit("GPUS 不能为空：请填写 uuid/name/serial（nvidia-smi 三列）")

    if not PRIVATE_KEY_PATH.is_file():
        raise SystemExit(f"私钥不存在，请先设 INIT_KEYS=True 运行: {PRIVATE_KEY_PATH}")
    if not Path(AES_KEY_PATH).is_file():
        raise SystemExit(f"模型密钥不存在，请先运行 convert_2_onnx.py: {AES_KEY_PATH}")

    aes_key, key_meta = load_key_file(AES_KEY_PATH)
    model_id = MODEL_ID or str(key_meta.get("model_id") or "")
    issued = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    payload = {
        "version": 2,
        "issued_at": issued,
        "expires_at": str(EXPIRES_AT or "").strip(),
        "customer": str(CUSTOMER or ""),
        "model_id": model_id,
        "note": str(NOTE or ""),
        "gpus": gpus,
        "aes_key_b64": base64.b64encode(aes_key).decode("ascii"),
    }
    private_key = load_private_key(PRIVATE_KEY_PATH)
    signed = sign_license(payload, private_key)
    if VERIFY_AFTER_SIGN:
        pub_path = _GO_ROOT / "internal" / "license" / "license_ed25519.pub"
        if not pub_path.is_file():
            pub_path = PUBLIC_KEY_PATH
        verify_license(signed, pub_path.read_bytes())
        logger.info("签名校验通过")
    save_json(Path(OUTPUT_PATH), signed)
    logger.info("已写入 License: %s", OUTPUT_PATH)
    logger.info("授权 GPU: %s", json.dumps(gpus, ensure_ascii=False))


if __name__ == "__main__":
    main()
