#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Detail  : 封边缺陷推理 Gradio 调试页 + FastAPI（本机路径 + multipart 上传）。
#           先注册 HTTP 再 mount Gradio，避免 launch 重建应用导致路由 404。

from __future__ import annotations

import asyncio
import json
import logging
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Union

import cv2
import gradio as gr
import numpy as np
from fastapi import Body, FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

_PREDICT_DIR = Path(__file__).resolve().parent
if str(_PREDICT_DIR) not in sys.path:
    sys.path.insert(0, str(_PREDICT_DIR))

from labels import labels_payload  # noqa: E402
from api_models import (  # noqa: E402
    CameraInput,
    HealthResponse,
    LabelsResponse,
    PREDICT_HTTP_RESPONSES,
    PREDICT_REQUEST_EXAMPLE,
    PREDICT_REQUEST_OBJECT_EXAMPLE,
    PredictRequest,
    PredictResponse,
    SchemeUpdate,
    SchemeView,
)
from predict_all import (  # noqa: E402
    DEFAULT_PREDICT_JSON,
    DefectPredictAll,
    draw_results,
    load_image_bgr,
    normalize_infer_type,
)
from scheme import (  # noqa: E402
    has_scheme_fields,
    incoming_scheme_overlay,
    index_defects,
    load_json,
    merge_scheme,
    scheme_public_view,
)

def dump_jsonable(value: Any) -> Any:
    if isinstance(value, list):
        return [dump_jsonable(x) for x in value]
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_unset=True)
    if hasattr(value, "dict"):
        return value.dict(exclude_unset=True)
    return value


def parse_predict_request(body: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    extras: dict[str, Any] = {}
    body = dump_jsonable(body)
    if isinstance(body, list):
        cameras: Any = body
    elif isinstance(body, dict):
        extras = body
        cameras = body.get("cameras")
        if cameras is None:
            cameras = body.get("items")
        if cameras is None and body.get("camera") is not None:
            cameras = [{"camera": body.get("camera"), "images": body.get("images")}]
    else:
        raise ValueError("请求body必须是JSON")
    if not isinstance(cameras, list) or not cameras:
        raise ValueError("请求body必须包含相机与图片路径")
    return cameras, extras


def _parse_form_bool(raw: str | None) -> bool | None:
    if raw is None or str(raw).strip() == "":
        return None
    text = str(raw).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"laminating无法解析: {raw}")


def parse_upload_cameras(
    *,
    n_files: int,
    meta_raw: str | None,
    camera: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """解析 multipart 元数据，返回 cameras 规格（含 count）与 extras。"""
    if n_files <= 0:
        raise ValueError("必须上传files")
    extras: dict[str, Any] = {}
    cameras_spec: Any = None
    if meta_raw is not None and str(meta_raw).strip():
        try:
            meta = json.loads(meta_raw)
        except json.JSONDecodeError as exc:
            raise ValueError("meta不是合法JSON") from exc
        if not isinstance(meta, dict):
            raise ValueError("meta必须是JSON对象")
        extras = dict(meta)
        cameras_spec = extras.pop("cameras", None)
        extras.pop("items", None)
        extras.pop("images", None)
    if cameras_spec is None:
        cam = str(camera or "ch1").strip() or "ch1"
        cameras_spec = [{"camera": cam, "count": n_files}]
    if not isinstance(cameras_spec, list) or not cameras_spec:
        raise ValueError("meta.cameras必须是非空数组")
    normalized: list[dict[str, Any]] = []
    total = 0
    for item in cameras_spec:
        if not isinstance(item, dict):
            raise ValueError("meta.cameras必须是非空数组")
        cam = str(item.get("camera") or "").strip()
        if not cam:
            raise ValueError("camera不能为空")
        try:
            count = int(item.get("count"))
        except (TypeError, ValueError) as exc:
            raise ValueError("count必须为正整数") from exc
        if count <= 0:
            raise ValueError("count必须为正整数")
        total += count
        normalized.append({"camera": cam, "count": count})
    if total != n_files:
        raise ValueError(
            f"上传文件数({n_files})与cameras.count合计({total})不一致"
        )
    return normalized, extras


def assign_upload_paths(
    cameras_spec: list[dict[str, Any]], paths: list[str]
) -> list[dict[str, Any]]:
    cameras: list[dict[str, Any]] = []
    idx = 0
    for item in cameras_spec:
        count = int(item["count"])
        cameras.append(
            {"camera": item["camera"], "images": paths[idx : idx + count]}
        )
        idx += count
    return cameras


logger = logging.getLogger(__name__)
_OUTPUT_DIR = Path(tempfile.gettempdir()) / "scanly_defects_outputs"
_UPLOAD_DIR = Path(tempfile.gettempdir()) / "scanly_defects_uploads"
_HEALTH_PATH = "/health/ready"


def _api_error(msg: str) -> JSONResponse:
    return JSONResponse({"code": 500, "msg": msg})


def _rgb_from_gradio(image: Any) -> np.ndarray | None:
    if image is None:
        return None
    if isinstance(image, dict):
        image = image.get("path") or image.get("name") or image.get("orig_name")
        if image is None:
            return None
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image
    path = Path(str(image).strip())
    if not path.is_file():
        return None
    bgr = load_image_bgr(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class DefectServingApp:
    def __init__(
        self,
        *,
        model_path: str | None = None,
        predict_cfg: dict[str, Any] | None = None,
        device: str | None = None,
        api_concurrency: int = 4,
    ) -> None:
        cfg = dict(predict_cfg or {})
        if not cfg and DEFAULT_PREDICT_JSON.is_file():
            cfg = load_json(DEFAULT_PREDICT_JSON)
        if model_path:
            cfg["model_path"] = model_path
        if device is not None:
            cfg["device"] = device
        self.predict_cfg = cfg
        self.pipeline = DefectPredictAll(
            model_path=cfg.get("model_path"),
            predict_cfg=cfg,
            device=device if device is not None else cfg.get("device"),
        )
        self._lock = threading.Lock()
        self._api_executor = ThreadPoolExecutor(
            max_workers=max(int(api_concurrency), 1),
            thread_name_prefix="scanly-api",
        )

    def is_ready(self) -> bool:
        return self.pipeline.is_ready()

    def readiness_payload(self) -> dict[str, Any]:
        return self.pipeline.readiness_payload()

    def predict_http(self, body: Any) -> dict[str, Any]:
        cameras, extras = parse_predict_request(body)
        calib = extras.get("calib") if isinstance(extras.get("calib"), dict) else {}
        mm_per_px = calib.get("mm_per_px")
        if mm_per_px is None:
            mm_per_px = extras.get("mm_per_px")
        overlay = extras.get("scheme") if isinstance(extras.get("scheme"), dict) else None
        if extras.get("scheme_name") and overlay is None:
            overlay = {"scheme_name": extras.get("scheme_name")}
        elif extras.get("scheme_name") and overlay is not None and "scheme_name" not in overlay:
            overlay = dict(overlay)
            overlay["scheme_name"] = extras.get("scheme_name")
        t0 = time.perf_counter()
        with self._lock:
            payload = self.pipeline.predict_cameras(
                cameras,
                board_id=str(extras.get("board_id") or ""),
                laminating=bool(extras.get("laminating") or False),
                mm_per_px=float(mm_per_px) if mm_per_px is not None else None,
                scheme_overlay=overlay,
                infer_type=extras.get("infer_type"),
            )
        payload["elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        return payload

    def update_scheme_http(self, body: dict[str, Any]) -> dict[str, Any]:
        persist = True if body.get("persist") is None else bool(body.get("persist"))
        replace = bool(body.get("replace") or False)
        overlay = incoming_scheme_overlay(body)
        if replace and not isinstance(body.get("defects"), list):
            raise ValueError("replace时必须提供defects数组")
        if not has_scheme_fields(overlay):
            raise ValueError("请求body无有效方案字段")
        with self._lock:
            return self.pipeline.apply_scheme_update(
                overlay, persist=persist, replace=replace
            )

    def run_gradio(
        self,
        image: Any,
        mm_per_px: float,
        laminating: bool,
        device: str,
        enabled_classes: list[str],
        infer_type: str,
    ) -> tuple[str | None, str, list[list[Any]], dict[str, Any], dict[str, Any]]:
        try:
            kind = normalize_infer_type(
                infer_type, default=self.pipeline.default_infer_type
            )
        except ValueError as exc:
            return None, str(exc), [], {}, {}
        if not self.pipeline.is_ready(kind):
            payload = self.readiness_payload()
            msg = "OBB模型未加载" if kind == "obb" else (payload.get("msg") or "模型未加载")
            return (
                None,
                f"**模型未加载**：{msg}。配置 `config/predict.json` 的 model_path / obb_model_path 后重启。GET /v1/scheme 仍可对接。",
                [],
                {},
                scheme_public_view(self.pipeline.base_scheme),
            )
        rgb = _rgb_from_gradio(image)
        if rgb is None:
            return None, "请先上传图片", [], {}, {}
        enabled = set(enabled_classes or [])
        overlay = {
            "defects": [
                {"name": name, "enable": name in enabled}
                for name in index_defects(self.pipeline.base_scheme)
            ]
        }
        scheme = merge_scheme(self.pipeline.base_scheme, overlay)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        use_device = None if str(device) in ("", "auto") else str(device)
        old_device = self.pipeline.device
        if use_device is not None:
            self.pipeline.device = use_device
        t0 = time.perf_counter()
        try:
            with self._lock:
                payload = self.pipeline.predict_bgr(
                    bgr,
                    laminating=bool(laminating),
                    mm_per_px=float(mm_per_px or 0.03),
                    scheme_overlay=overlay,
                    infer_type=kind,
                )
        finally:
            self.pipeline.device = old_device
        payload["elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        drawn = draw_results(bgr, payload.get("results") or [])
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = _OUTPUT_DIR / f"result_{uuid.uuid4().hex}.jpg"
        cv2.imwrite(str(out_path), drawn, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        rows = []
        for item in payload.get("results") or []:
            loc = item.get("location") or {}
            rows.append(
                [
                    item.get("cn_name") or "",
                    item.get("name") or "",
                    item.get("score"),
                    item.get("height_mm"),
                    item.get("length_mm"),
                    bool(item.get("ng")),
                    bool(item.get("need_label")),
                    bool(item.get("need_discharge")),
                    loc.get("left"),
                    loc.get("top"),
                    loc.get("width"),
                    loc.get("height"),
                ]
            )
        summary = (
            f"检出 **{len(rows)}** 条，NG **{payload.get('ng_count')}**，"
            f"贴标 **{payload.get('label_count')}**，排出 **{payload.get('discharge_count')}**，"
            f"infer_type={payload.get('infer_type')}，image_ok={payload.get('image_ok')}，"
            f"耗时 {payload.get('elapsed_ms')} ms。模型就绪={self.pipeline.is_ready(kind)}。"
        )
        # 本次检出结果（与上表一致）；方案配置单独一栏，避免误当成检出列表
        result_json = {
            "infer_type": payload.get("infer_type"),
            "image_ok": payload.get("image_ok"),
            "ng_count": payload.get("ng_count"),
            "label_count": payload.get("label_count"),
            "discharge_count": payload.get("discharge_count"),
            "elapsed_ms": payload.get("elapsed_ms"),
            "counts": payload.get("counts") or {},
            "results": payload.get("results") or [],
        }
        return (
            str(out_path),
            summary,
            rows,
            result_json,
            scheme_public_view(scheme),
        )

    def build_ui(self) -> gr.Blocks:
        defaults = index_defects(self.pipeline.base_scheme)
        choices = []
        selected = []
        for name, item in defaults.items():
            cn = str(item.get("cn_name") or name)
            choices.append((f"{cn} | {name}", name))
            if item.get("enable", True):
                selected.append(name)
        mm_default = float(
            self.pipeline.base_scheme.get("mm_per_px") or self.pipeline.default_mm_per_px
        )
        title = "封边缺陷检测调试（scanly-defects-v1）"
        with gr.Blocks(title=title) as demo:
            gr.Markdown(
                f"## {title}\n"
                "上传本机图片调试；产线请调 `POST /v1/predict`（路径）"
                "或 `POST /v1/predict/upload`（multipart），"
                "文档见 `scanly/doc/02-dr/缺陷推理/01.HTTP接口文档.md`。"
            )
            with gr.Row():
                with gr.Column():
                    image_in = gr.Image(type="filepath", label="图片（可拖拽本地文件）")
                    mm_in = gr.Number(value=mm_default, label="mm_per_px")
                    laminating_in = gr.Checkbox(value=False, label="覆膜板（laminating）")
                    infer_type_in = gr.Dropdown(
                        choices=["detect", "obb"],
                        value=self.pipeline.default_infer_type,
                        label="推理类型 infer_type",
                    )
                    device_in = gr.Dropdown(
                        choices=["auto", "cpu", "cuda:0"],
                        value="auto",
                        label="设备",
                    )
                    classes_in = gr.CheckboxGroup(
                        choices=choices,
                        value=selected,
                        label="启用的缺陷类别",
                    )
                    run_btn = gr.Button("运行识别", variant="primary")
                with gr.Column():
                    image_out = gr.Image(label="结果图", type="filepath")
            summary_out = gr.Markdown()
            table_out = gr.Dataframe(
                headers=[
                    "中文",
                    "name",
                    "score",
                    "height_mm",
                    "length_mm",
                    "ng",
                    "贴标",
                    "排出",
                    "left",
                    "top",
                    "width",
                    "height",
                ],
                label="本次检出结果（表格）",
            )
            result_out = gr.JSON(label="本次检出结果 JSON（与上表同一批缺陷，不是方案配置）")
            scheme_out = gr.JSON(
                label="判定方案配置（enable/阈值等；defects 是类别参数表，不是本次检出列表）"
            )
            run_btn.click(
                fn=self.run_gradio,
                inputs=[image_in, mm_in, laminating_in, device_in, classes_in, infer_type_in],
                outputs=[image_out, summary_out, table_out, result_out, scheme_out],
            )
        return demo


def register_http_routes(fastapi_app: FastAPI, svc: DefectServingApp) -> None:
    @fastapi_app.get(
        _HEALTH_PATH,
        summary="就绪检查",
        response_model=HealthResponse,
        responses={503: {"model": HealthResponse, "description": "模型未加载"}},
    )
    async def health_ready() -> Any:
        payload = svc.readiness_payload()
        status = 200 if payload.get("ready") else 503
        return JSONResponse(payload, status_code=status)

    @fastapi_app.get(
        "/v1/labels",
        summary="查询缺陷标签对照表",
        response_model=LabelsResponse,
    )
    async def get_labels() -> Any:
        return JSONResponse(
            {"code": 0, "engine": "scanly-defects-v1", "items": labels_payload()}
        )

    @fastapi_app.get(
        "/v1/scheme",
        summary="查询方案参数",
        response_model=SchemeView,
    )
    async def get_scheme() -> Any:
        return JSONResponse(svc.pipeline.scheme_view())

    async def _apply_scheme_body(body: dict[str, Any]) -> Any:
        if not isinstance(body, dict):
            return _api_error("请求body必须是JSON")
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                svc._api_executor, svc.update_scheme_http, body
            )
            logger.info(
                "[API] scheme updated persist=%s replace=%s name=%s",
                result.get("persisted"),
                body.get("replace"),
                result.get("scheme_name"),
            )
            return JSONResponse(result)
        except ValueError as exc:
            return _api_error(str(exc))
        except RuntimeError as exc:
            logger.exception("[API] 写入方案失败")
            return _api_error(str(exc) or "写入方案文件失败")

    @fastapi_app.put(
        "/v1/scheme",
        summary="配置方案参数",
        response_model=SchemeView,
    )
    async def put_scheme(body: SchemeUpdate) -> Any:
        return await _apply_scheme_body(dump_jsonable(body))

    @fastapi_app.post(
        "/v1/scheme",
        summary="配置方案参数（与 PUT 相同）",
        response_model=SchemeView,
    )
    async def post_scheme(body: SchemeUpdate) -> Any:
        return await _apply_scheme_body(dump_jsonable(body))

    async def _run_predict(body: Any) -> Any:
        dumped = dump_jsonable(body)
        try:
            _, extras = parse_predict_request(dumped)
            kind = normalize_infer_type(
                extras.get("infer_type"),
                default=svc.pipeline.default_infer_type,
            )
        except ValueError as exc:
            return _api_error(str(exc))
        if not svc.pipeline.is_ready(kind):
            if kind == "obb":
                return _api_error("OBB模型未加载")
            if extras.get("infer_type"):
                return _api_error("检测模型未加载")
            return _api_error("推理服务尚未就绪，请稍后重试")
        logger.info("[API] predict cameras body=%s", body)
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(svc._api_executor, svc.predict_http, dumped)
            n_def = sum(
                len(slot)
                for cam in (result.get("results") or [])
                for slot in (cam.get("defects") or [])
            )
            logger.info(
                "[API] predict done cameras=%d defects=%d elapsed_ms=%s",
                len(result.get("results") or []),
                n_def,
                result.get("elapsed_ms"),
            )
            return JSONResponse(result)
        except ValueError as exc:
            logger.warning("[API] 读图失败: %s", exc)
            text = str(exc)
            if any(x in text for x in ("不存在", "解码", "图片")):
                return _api_error("读取图片异常")
            return _api_error(text)
        except RuntimeError as exc:
            text = str(exc)
            if "模型未加载" in text:
                return _api_error(text)
            logger.exception("[API] 预测失败")
            return _api_error("读取图片异常")
        except Exception:
            logger.exception("[API] 预测失败")
            return _api_error("读取图片异常")

    @fastapi_app.post(
        "/v1/predict",
        summary="按相机推理",
        response_model=PredictResponse,
        responses=PREDICT_HTTP_RESPONSES,
    )
    async def v1_predict(
        body: Union[PredictRequest, list[CameraInput]] = Body(
            ...,
            description="相机数组，或 {cameras, infer_type} 对象。defects[i] 对应 images[i]",
            openapi_examples={
                "cameras": {
                    "summary": "相机数组（使用配置默认 infer_type）",
                    "value": PREDICT_REQUEST_EXAMPLE,
                },
                "obb": {
                    "summary": "指定 OBB 推理",
                    "value": PREDICT_REQUEST_OBJECT_EXAMPLE,
                },
            },
        ),
    ) -> Any:
        return await _run_predict(dump_jsonable(body))

    @fastapi_app.post(
        "/scanly_predict",
        summary="按相机推理（与 /v1/predict 相同）",
        response_model=PredictResponse,
        responses=PREDICT_HTTP_RESPONSES,
    )
    async def scanly_predict(
        body: Union[PredictRequest, list[CameraInput]] = Body(
            ...,
            description="相机数组，或 {cameras, infer_type} 对象。defects[i] 对应 images[i]",
            openapi_examples={
                "cameras": {
                    "summary": "相机数组（使用配置默认 infer_type）",
                    "value": PREDICT_REQUEST_EXAMPLE,
                },
                "obb": {
                    "summary": "指定 OBB 推理",
                    "value": PREDICT_REQUEST_OBJECT_EXAMPLE,
                },
            },
        ),
    ) -> Any:
        return await _run_predict(dump_jsonable(body))

    @fastapi_app.post(
        "/v1/predict/upload",
        summary="按相机推理（multipart 上传图像）",
        response_model=PredictResponse,
        responses=PREDICT_HTTP_RESPONSES,
    )
    async def v1_predict_upload(
        files: list[UploadFile] = File(
            ..., description="一张或多张图像，顺序与 meta.cameras[].count 对齐"
        ),
        meta: str | None = Form(
            None,
            description=(
                '可选 JSON：{"cameras":[{"camera":"ch1","count":1}],'
                '"infer_type":"obb",...}'
            ),
        ),
        camera: str | None = Form(
            None, description="未写 meta.cameras 时，全部 files 归该相机；缺省 ch1"
        ),
        infer_type: str | None = Form(None, description="detect 或 obb，覆盖 meta"),
        laminating: str | None = Form(None, description="true/false"),
        mm_per_px: float | None = Form(None),
        board_id: str | None = Form(None),
    ) -> Any:
        saved: list[Path] = []
        try:
            if not files:
                return _api_error("必须上传files")
            try:
                cameras_spec, extras = parse_upload_cameras(
                    n_files=len(files), meta_raw=meta, camera=camera
                )
            except ValueError as exc:
                return _api_error(str(exc))
            if infer_type is not None and str(infer_type).strip():
                extras["infer_type"] = str(infer_type).strip()
            if board_id is not None:
                extras["board_id"] = board_id
            try:
                lam = _parse_form_bool(laminating)
            except ValueError as exc:
                return _api_error(str(exc))
            if lam is not None:
                extras["laminating"] = lam
            if mm_per_px is not None:
                extras["mm_per_px"] = float(mm_per_px)

            _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            for upload in files:
                data = await upload.read()
                if not data:
                    return _api_error("上传文件为空")
                suffix = Path(upload.filename or "img.bin").suffix or ".bin"
                path = _UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
                path.write_bytes(data)
                saved.append(path)

            cameras = assign_upload_paths(
                cameras_spec, [str(p) for p in saved]
            )
            body: dict[str, Any] = {**extras, "cameras": cameras}
            logger.info(
                "[API] predict/upload cameras=%s files=%d",
                [c.get("camera") for c in cameras],
                len(saved),
            )
            return await _run_predict(body)
        finally:
            for path in saved:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("[API] 清理上传临时文件失败: %s", path)


def configure_process_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.StreamHandler(sys.stderr)],
        force=True,
    )
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).setLevel(level)


def create_serving_app(
    *,
    model_path: str | None = None,
    predict_cfg: dict[str, Any] | None = None,
    device: str | None = None,
    mount_path: str = "/",
) -> FastAPI:
    svc = DefectServingApp(
        model_path=model_path, predict_cfg=predict_cfg, device=device
    )
    demo = svc.build_ui().queue(default_concurrency_limit=2)
    fastapi_app = FastAPI(
        title="封边缺陷推理",
        version="scanly-defects-v1",
        description=(
            "边端 HTTP：本机图片路径（/v1/predict）或 multipart 上传（/v1/predict/upload）。"
            "infer_type=detect|obb。契约见 scanly/doc/02-dr/缺陷推理/01.HTTP接口文档.md"
        ),
    )
    register_http_routes(fastapi_app, svc)
    return gr.mount_gradio_app(fastapi_app, demo, path=mount_path)


if __name__ == "__main__":
    configure_process_logging(logging.INFO)

    PREDICT_JSON: str | None = None
    MODEL_PATH: str | None = None
    OBB_MODEL_PATH: str | None = None
    INFER_TYPE: str | None = None
    SERVER_NAME = "0.0.0.0"
    SERVER_PORT = 37870
    INFER_DEVICE: str | None = None

    import uvicorn

    cfg_path = Path(PREDICT_JSON) if PREDICT_JSON else DEFAULT_PREDICT_JSON
    cfg = load_json(cfg_path) if cfg_path.is_file() else {}
    if OBB_MODEL_PATH:
        cfg["obb_model_path"] = OBB_MODEL_PATH
    if INFER_TYPE:
        cfg["infer_type"] = INFER_TYPE
    host = str(SERVER_NAME or cfg.get("host") or "0.0.0.0")
    port = int(SERVER_PORT or cfg.get("port") or 37870)
    logger.info(
        "启动服务 %s:%s（Gradio + POST /v1/predict + /v1/predict/upload，detect=%s obb=%s infer_type=%s）",
        host,
        port,
        MODEL_PATH or cfg.get("model_path") or "(未配置)",
        cfg.get("obb_model_path") or "(未配置)",
        cfg.get("infer_type") or "obb",
    )
    app = create_serving_app(
        model_path=MODEL_PATH, predict_cfg=cfg, device=INFER_DEVICE
    )
    uvicorn.run(app, host=host, port=port, log_config=None)
