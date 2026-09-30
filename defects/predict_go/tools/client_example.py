#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Detail  : Go 推理服务 HTTP 试调。改下方变量后在 IDE 运行。

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib import error, request

BASE_URL = "http://127.0.0.1:37871"
PREDICT_PATH = "/v1/predict"
INFER_TYPE = ""
CAMERAS: list[dict[str, Any]] = [
    {"camera": "ch1", "images": []},
]


def request_json(url: str, body: Any | None = None, *, method: str = "GET", timeout: float = 120.0) -> tuple[int, dict[str, Any]]:
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    req = request.Request(url, data=data, headers=headers, method=method)
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
    base = BASE_URL.rstrip("/")
    print("root:", request_json(f"{base}/"))
    print("health:", request_json(f"{base}/health/ready"))
    print("license:", request_json(f"{base}/v1/license"))
    print("labels nc=", len(request_json(f"{base}/v1/labels")[1].get("items") or []))
    print("scheme:", request_json(f"{base}/v1/scheme")[1].get("scheme_name"))

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
        status, payload = request_json(
            f"{base}{PREDICT_PATH}",
            {"infer_type": INFER_TYPE, "cameras": CAMERAS} if INFER_TYPE else CAMERAS,
            method="POST",
        )
        print("HTTP", status)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
