#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Detail  : 调用缺陷推理 HTTP API：服务端本机图片路径检测（不上传）。
#           适用于 Go 服务（默认 37871）或 Python 服务的 POST /v1/predict。
#           改下方 __main__ 变量后在 IDE 运行。
#           注意：IMAGE_PATHS 必须是「推理服务所在机器」上可读的绝对路径。

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from urllib import error, request

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


def post_json(
    url: str,
    body: Any,
    *,
    timeout: float = 180.0,
) -> tuple[int, dict[str, Any]]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
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


def run_path_detect(
    *,
    base_url: str,
    cameras: list[dict[str, Any]],
    infer_type: str = "",
    check_ready: bool = True,
    check_local_files: bool = False,
) -> dict[str, Any]:
    """cameras: [{"camera": "ch1", "images": ["/server/path/a.jpg"]}, ...]"""
    base = base_url.rstrip("/")
    if not cameras:
        raise ValueError("cameras 不能为空")

    expect_cameras: list[str] = []
    expect_counts: list[int] = []
    for cam in cameras:
        name = str(cam.get("camera") or "").strip()
        images = [str(p).strip() for p in (cam.get("images") or []) if str(p).strip()]
        if not name:
            raise ValueError("camera 不能为空")
        if not images:
            raise ValueError(f"相机 {name} 的 images 不能为空")
        expect_cameras.append(name)
        expect_counts.append(len(images))
        if check_local_files:
            for p in images:
                if not Path(p).is_file():
                    raise FileNotFoundError(f"本机找不到图片（仅当测试机=服务机时检查）: {p}")

    if check_ready:
        health_status, health = get_json(f"{base}/health/ready")
        print("health:", health_status, health)
        if health_status != 200 or not health.get("ready"):
            raise RuntimeError(
                f"服务未就绪（HTTP {health_status}）：{health.get('msg') or health}"
            )

    body: dict[str, Any] = {"cameras": cameras}
    if str(infer_type or "").strip():
        body["infer_type"] = str(infer_type).strip()

    status, payload = post_json(f"{base}/v1/predict", body)
    print("predict HTTP", status)
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    assert_predict_ok(
        status,
        payload,
        expect_cameras=expect_cameras,
        expect_counts=expect_counts,
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
    # Go 默认 37871；Python 服务为 37870
    BASE_URL = "http://36.133.115.106:37871"
    INFER_TYPE = "obb"  # detect | obb；空字符串则用服务配置默认
    CAMERA = "ch1"
    # 必须是推理服务所在机器上的路径；远程调用时不要用本机 Mac 路径
    # IMAGE_PATHS: list[str] = [
    #     str(_DEFAULT_SAMPLE) if _DEFAULT_SAMPLE else "",
    # ]
    # 示例：
    IMAGE_PATHS = [r"/root/scanly/0ac5d279-2025-11-22_09_00_53.860_WD000618102E1191_3_2_10.jpg"]
    # 多相机：
    # CAMERAS = [
    #     {"camera": "ch1", "images": [r"/data/a.jpg"]},
    #     {"camera": "ch2", "images": [r"/data/b.jpg"]},
    # ]
    CAMERAS: list[dict[str, Any]] | None = None
    CHECK_READY = True
    # 仅当本机就是服务机、且路径在本机存在时设 True
    CHECK_LOCAL_FILES = False

    if CAMERAS is None:
        paths = [p for p in IMAGE_PATHS if str(p).strip()]
        if not paths:
            raise SystemExit(
                "未配置 IMAGE_PATHS。请填写服务端本机图片绝对路径，"
                f"或把样张放到 {_DEFECTS_DIR / 'samples'} 且 CHECK_LOCAL_FILES=True"
            )
        CAMERAS = [{"camera": CAMERA, "images": paths}]

    try:
        run_path_detect(
            base_url=BASE_URL,
            cameras=CAMERAS,
            infer_type=INFER_TYPE,
            check_ready=CHECK_READY,
            check_local_files=CHECK_LOCAL_FILES,
        )
    except Exception as exc:
        print(f"失败: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
