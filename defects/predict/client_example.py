#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Detail  : 封边缺陷 HTTP 试调。改下方变量后在 IDE 运行。
#           MODE=path 走 JSON 路径；MODE=upload 走 multipart 上传。

from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Any
from urllib import error, request
from uuid import uuid4

BASE_URL = "http://127.0.0.1:37870"
PREDICT_PATH = "/v1/predict"
UPLOAD_PATH = "/v1/predict/upload"
MODE = "path"  # path | upload
INFER_TYPE = ""  # detect 或 obb；空字符串则用服务配置默认（obb）
# 路径模式：按相机分组的本机图像路径
CAMERAS: list[dict[str, Any]] = [
    {"camera": "ch1", "images": []},  # 填绝对路径，如 r"D:\edge\a.bmp"
]
# 上传模式：本地文件列表；CAMERA 为单路简写；多相机用 UPLOAD_META
UPLOAD_FILES: list[str] = []  # 如 [r"/data/a.bmp"]
CAMERA = "ch1"
UPLOAD_META: dict[str, Any] | None = None
# 多相机示例：
# UPLOAD_META = {"cameras": [{"camera": "ch1", "count": 2}, {"camera": "ch2", "count": 1}]}
# UPLOAD_FILES = [r"/data/c1_1.bmp", r"/data/c1_2.bmp", r"/data/c2_1.bmp"]


def request_json(
    url: str,
    body: Any,
    *,
    method: str = "POST",
    timeout: float = 120.0,
) -> tuple[int, dict[str, Any]]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method=method,
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            status = resp.status
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        status = exc.code
    parsed: dict[str, Any] = json.loads(raw) if raw else {}
    return status, parsed


def post_json(url: str, body: Any, timeout: float = 120.0) -> tuple[int, dict[str, Any]]:
    return request_json(url, body, method="POST", timeout=timeout)


def get_json(url: str, timeout: float = 30.0) -> tuple[int, dict[str, Any]]:
    req = request.Request(url, method="GET")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            status = resp.status
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        status = exc.code
    parsed: dict[str, Any] = json.loads(raw) if raw else {}
    return status, parsed


def post_multipart_upload(
    url: str,
    files: list[Path],
    *,
    camera: str | None = None,
    meta: dict[str, Any] | None = None,
    infer_type: str = "",
    timeout: float = 120.0,
) -> tuple[int, dict[str, Any]]:
    boundary = f"----scanly{uuid4().hex}"
    parts: list[bytes] = []

    def add_field(name: str, value: str) -> None:
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode("utf-8")
        )

    if meta is not None:
        add_field("meta", json.dumps(meta, ensure_ascii=False))
    elif camera:
        add_field("camera", camera)
    if infer_type:
        add_field("infer_type", infer_type)

    for path in files:
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        header = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="files"; filename="{path.name}"\r\n'
            f"Content-Type: {mime}\r\n\r\n"
        ).encode("utf-8")
        parts.append(header + path.read_bytes() + b"\r\n")

    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(parts)
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            status = resp.status
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        status = exc.code
    parsed: dict[str, Any] = json.loads(raw) if raw else {}
    return status, parsed


if __name__ == "__main__":
    print("health:", get_json(f"{BASE_URL.rstrip('/')}/health/ready"))
    print("labels nc=", len(get_json(f"{BASE_URL.rstrip('/')}/v1/labels")[1].get("items") or []))
    scheme_status, scheme_body = get_json(f"{BASE_URL.rstrip('/')}/v1/scheme")
    print("scheme:", scheme_status, scheme_body.get("scheme_name"))

    UPDATE_SCHEME = False
    if UPDATE_SCHEME:
        status, payload = request_json(
            f"{BASE_URL.rstrip('/')}/v1/scheme",
            {"persist": False, "defects": [{"name": "Tackless", "confidence": 0.5}]},
            method="PUT",
        )
        print("PUT scheme HTTP", status, payload.get("msg"), "persisted=", payload.get("persisted"))

    if MODE == "upload":
        paths = [Path(p) for p in UPLOAD_FILES if str(p).strip()]
        if not paths:
            print("未设置 UPLOAD_FILES，跳过上传预测。")
        else:
            missing = [p for p in paths if not p.is_file()]
            if missing:
                raise SystemExit(f"图片不存在: {missing[0]}")
            status, payload = post_multipart_upload(
                f"{BASE_URL.rstrip('/')}{UPLOAD_PATH}",
                paths,
                camera=None if UPLOAD_META else CAMERA,
                meta=UPLOAD_META,
                infer_type=INFER_TYPE,
            )
            print("HTTP", status)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        has_image = any(str(p).strip() for cam in CAMERAS for p in (cam.get("images") or []))
        if not has_image:
            print("未设置 CAMERAS[].images，跳过预测。")
        else:
            missing = [
                p
                for cam in CAMERAS
                for p in (cam.get("images") or [])
                if str(p).strip() and not Path(str(p)).is_file()
            ]
            if missing:
                raise SystemExit(f"图片不存在: {missing[0]}")
            status, payload = post_json(
                f"{BASE_URL.rstrip('/')}{PREDICT_PATH}",
                {"infer_type": INFER_TYPE, "cameras": CAMERAS} if INFER_TYPE else CAMERAS,
            )
            print("HTTP", status)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
