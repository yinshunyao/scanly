#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Detail  : 调用缺陷推理 HTTP API：上传本地图片并检测。
#           需先启动 serve.py（默认 http://127.0.0.1:37870）。
#           改下方 __main__ 变量后在 IDE 运行。

from __future__ import annotations

import json
import mimetypes
import sys
from pathlib import Path
from typing import Any
from urllib import error, request
from uuid import uuid4

_TEST_DIR = Path(__file__).resolve().parent
_DEFECTS_DIR = _TEST_DIR.parent
_DEFAULT_SAMPLE = next(
    (
        p
        for p in sorted((_DEFECTS_DIR / "samples").glob("*.bmp"))
        if p.is_file()
    ),
    None,
)


def get_json(url: str, timeout: float = 30.0) -> tuple[int, dict[str, Any]]:
    req = request.Request(url, method="GET")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            status = int(resp.status)
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        status = int(exc.code)
    except error.URLError as exc:
        raise RuntimeError(f"无法连接服务 {url}: {exc}") from exc
    parsed: dict[str, Any] = json.loads(raw) if raw else {}
    return status, parsed


def post_multipart_upload(
    url: str,
    files: list[Path],
    *,
    camera: str | None = None,
    meta: dict[str, Any] | None = None,
    infer_type: str = "",
    laminating: bool | None = None,
    mm_per_px: float | None = None,
    board_id: str = "",
    timeout: float = 180.0,
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
    if laminating is not None:
        add_field("laminating", "true" if laminating else "false")
    if mm_per_px is not None:
        add_field("mm_per_px", str(mm_per_px))
    if board_id:
        add_field("board_id", board_id)

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
            status = int(resp.status)
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        status = int(exc.code)
    except error.URLError as exc:
        raise RuntimeError(f"无法连接服务 {url}: {exc}") from exc
    parsed: dict[str, Any] = json.loads(raw) if raw else {}
    return status, parsed


def assert_predict_ok(
    status: int,
    payload: dict[str, Any],
    *,
    expect_cameras: list[str],
    expect_counts: list[int],
) -> None:
    if status != 200:
        raise AssertionError(f"HTTP 非 200: {status}, body={payload}")
    if int(payload.get("code", -1)) != 0:
        raise AssertionError(f"业务失败 code={payload.get('code')} msg={payload.get('msg')}")
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != len(expect_cameras):
        raise AssertionError(
            f"results 相机数不符: got={len(results) if isinstance(results, list) else results}, "
            f"expect={len(expect_cameras)}"
        )
    for i, (cam, n_img) in enumerate(zip(expect_cameras, expect_counts)):
        slot = results[i]
        if slot.get("camera") != cam:
            raise AssertionError(f"results[{i}].camera={slot.get('camera')!r}, expect={cam!r}")
        defects = slot.get("defects")
        if not isinstance(defects, list) or len(defects) != n_img:
            raise AssertionError(
                f"results[{i}].defects 长度不符: got={len(defects) if isinstance(defects, list) else defects}, "
                f"expect={n_img}"
            )
        for j, items in enumerate(defects):
            if not isinstance(items, list):
                raise AssertionError(f"defects[{j}] 应为列表，got={type(items)}")


def run_upload_detect(
    *,
    base_url: str,
    image_paths: list[Path],
    camera: str = "ch1",
    meta: dict[str, Any] | None = None,
    infer_type: str = "",
    check_ready: bool = True,
) -> dict[str, Any]:
    base = base_url.rstrip("/")
    paths = [Path(p) for p in image_paths]
    if not paths:
        raise ValueError("image_paths 不能为空")
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"图片不存在: {missing[0]}")

    if check_ready:
        health_status, health = get_json(f"{base}/health/ready")
        print("health:", health_status, health)
        if health_status != 200 or not health.get("ready"):
            raise RuntimeError(
                f"服务未就绪（HTTP {health_status}）：{health.get('msg') or health}"
            )

    status, payload = post_multipart_upload(
        f"{base}/v1/predict/upload",
        paths,
        camera=None if meta else camera,
        meta=meta,
        infer_type=infer_type,
    )
    print("upload HTTP", status)
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if meta and isinstance(meta.get("cameras"), list):
        expect_cameras = [str(c.get("camera") or "") for c in meta["cameras"]]
        expect_counts = [int(c.get("count") or 0) for c in meta["cameras"]]
    else:
        expect_cameras = [camera]
        expect_counts = [len(paths)]
    assert_predict_ok(
        status, payload, expect_cameras=expect_cameras, expect_counts=expect_counts
    )

    n_def = sum(
        len(items)
        for cam in (payload.get("results") or [])
        for items in (cam.get("defects") or [])
    )
    print(
        f"通过: infer_type={payload.get('infer_type')}, "
        f"defects={n_def}, elapsed_ms={payload.get('elapsed_ms')}"
    )
    return payload


if __name__ == "__main__":
    BASE_URL = "http://36.133.115.106:37870"
    INFER_TYPE = "obb"  # detect | obb；空字符串则用服务配置默认
    CAMERA = "ch1"
    # 默认用 defects/samples 下第一张 bmp；也可改成本地绝对路径列表
    IMAGE_PATHS: list[str] = [
        str(_DEFAULT_SAMPLE) if _DEFAULT_SAMPLE else "",
    ]

    IMAGE_PATHS =[r"/Users/shunyaoyin/Documents/code/ai-company/scanly/样本数据/cam123-0820/Bumps/images/1d3fba0e-62280_2025-10-23_16_30_54.990_WD000612891A1013_2_3_4.jpg"]
    # 多相机示例（同时设置 META，IMAGE_PATHS 顺序与 count 对齐）：
    # META = {"cameras": [{"camera": "ch1", "count": 1}, {"camera": "ch2", "count": 1}]}
    # IMAGE_PATHS = [r"/path/a.bmp", r"/path/b.bmp"]
    META: dict[str, Any] | None = None
    CHECK_READY = True

    paths = [Path(p) for p in IMAGE_PATHS if str(p).strip()]
    if not paths:
        raise SystemExit(
            "未配置 IMAGE_PATHS。请填写本地图片路径，"
            f"或把样张放到 {_DEFECTS_DIR / 'samples'}"
        )

    try:
        run_upload_detect(
            base_url=BASE_URL,
            image_paths=paths,
            camera=CAMERA,
            meta=META,
            infer_type=INFER_TYPE,
            check_ready=CHECK_READY,
        )
    except Exception as exc:
        print(f"失败: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
