#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""OpenAPI / Swagger 用的请求与响应模型（与 HTTP 契约一致）。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

PREDICT_REQUEST_EXAMPLE: list[dict[str, Any]] = [
    {
        "camera": "ch1",
        "images": ["C:/data/c1_pic1.bmp", "C:/data/c1_pic2.bmp"],
    },
    {"camera": "ch2", "images": ["C:/data/c2_pic1.bmp"]},
]

_DEFECT_EXAMPLE = {
    "name": "GlueSeam",
    "cn_name": "胶缝",
    "score": 0.86,
    "infer_type": "detect",
    "location": {"left": 120, "top": 40, "width": 180, "height": 8},
    "obb": {
        "points": [[120, 40], [300, 40], [300, 48], [120, 48]],
        "cx": 210.0,
        "cy": 44.0,
        "width": 180.0,
        "height": 8.0,
        "angle": 0.0,
    },
    "height_mm": 0.24,
    "length_mm": 5.4,
    "area_mm2": 1.3,
    "ng": True,
    "need_label": True,
    "need_discharge": False,
}

_DEFECT_OBB_EXAMPLE = {
    "name": "Shorter",
    "cn_name": "短带",
    "score": 0.86,
    "infer_type": "obb",
    "location": {"left": 120, "top": 40, "width": 180, "height": 24},
    "obb": {
        "points": [[120, 42], [298, 40], [300, 62], [122, 64]],
        "cx": 210.0,
        "cy": 52.0,
        "width": 180.2,
        "height": 22.1,
        "angle": 0.02,
    },
    "height_mm": 0.66,
    "length_mm": 5.41,
    "area_mm2": 3.57,
    "ng": True,
    "need_label": True,
    "need_discharge": False,
}

PREDICT_SUCCESS_EXAMPLE: dict[str, Any] = {
    "code": 0,
    "msg": "ok",
    "engine": "scanly-defects-v1",
    "infer_type": "obb",
    "board_id": "",
    "board_ok": False,
    "laminating": False,
    "elapsed_ms": 82.1,
    "results": [
        {
            "camera": "ch1",
            "defects": [[_DEFECT_OBB_EXAMPLE], []],
        },
        {"camera": "ch2", "defects": [[]]},
    ],
}

PREDICT_REQUEST_OBJECT_EXAMPLE: dict[str, Any] = {
    "cameras": [
        {"camera": "ch1", "images": ["C:/data/c1_pic1.bmp"]},
    ],
    "laminating": False,
    "calib": {"mm_per_px": 0.03},
}

PREDICT_SUCCESS_OBB_EXAMPLE: dict[str, Any] = {
    "code": 0,
    "msg": "ok",
    "engine": "scanly-defects-v1",
    "infer_type": "obb",
    "board_id": "",
    "board_ok": False,
    "laminating": False,
    "elapsed_ms": 82.1,
    "results": [
        {"camera": "ch1", "defects": [[_DEFECT_OBB_EXAMPLE]]},
    ],
}

API_ERROR_EXAMPLE: dict[str, Any] = {"code": 500, "msg": "读取图片异常"}


class CameraInput(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": PREDICT_REQUEST_EXAMPLE[0]})

    camera: str = Field(..., description="相机号，由边端定义，例如 ch1")
    images: list[str] = Field(
        ...,
        description="本机图像绝对路径列表；顺序与返回 defects 按下标对齐；无图传 []",
        examples=[["C:/data/c1_pic1.bmp", "C:/data/c1_pic2.bmp"]],
    )


class UploadCameraSpec(BaseModel):
    camera: str = Field(..., description="相机号")
    count: int = Field(..., ge=1, description="该路相机占用后续 files 的张数")


PREDICT_UPLOAD_META_EXAMPLE: dict[str, Any] = {
    "cameras": [
        {"camera": "ch1", "count": 2},
        {"camera": "ch2", "count": 1},
    ],
    "infer_type": "obb",
    "laminating": False,
    "calib": {"mm_per_px": 0.03},
}


class PredictRequest(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={"example": PREDICT_REQUEST_OBJECT_EXAMPLE},
    )

    cameras: list[CameraInput] = Field(..., description="按相机分组的本机图片路径")
    infer_type: str | None = Field(
        None,
        description="detect（轴对齐）或 obb（旋转框）；缺省用 predict.json，配置未写时默认 obb",
        examples=["obb"],
    )
    board_id: str = ""
    laminating: bool = False
    mm_per_px: float | None = None
    calib: dict[str, Any] | None = None
    scheme: dict[str, Any] | None = None
    scheme_name: str | None = None


class BBoxLocation(BaseModel):
    left: int = Field(..., description="像素，左上角 x")
    top: int = Field(..., description="像素，左上角 y")
    width: int = Field(..., description="像素宽")
    height: int = Field(..., description="像素高")


class ObbGeometry(BaseModel):
    points: list[list[int]] = Field(..., description="四点像素坐标 [[x,y],...]")
    cx: float = Field(..., description="旋转框中心 x")
    cy: float = Field(..., description="旋转框中心 y")
    width: float = Field(..., description="旋转框宽（像素）")
    height: float = Field(..., description="旋转框高（像素）")
    angle: float = Field(..., description="旋转角，弧度")


class DefectItem(BaseModel):
    name: str = Field(..., description="AI 标签，如 GlueSeam")
    cn_name: str = Field("", description="中文名，如 胶缝")
    score: float = Field(..., description="置信度")
    infer_type: str = Field("obb", description="detect 或 obb；缺省与服务默认一致")
    location: BBoxLocation
    obb: ObbGeometry | None = Field(
        None,
        description="统一 OBB 输出：四点与 xywhr；detect 时为矩形四顶点且 angle=0",
    )
    height_mm: float = Field(..., description="有效短边 × mm_per_px，对应客户「高度」")
    length_mm: float = Field(..., description="有效长边 × mm_per_px，对应客户「长度」")
    area_mm2: float = Field(..., description="框面积 mm²")
    ng: bool = Field(..., description="是否达到 NG 尺寸规则")
    need_label: bool = Field(..., description="是否达到贴标规则")
    need_discharge: bool = Field(..., description="是否达到排出规则")


class CameraDefects(BaseModel):
    camera: str = Field(..., description="回显入参相机号")
    defects: list[list[DefectItem]] = Field(
        ...,
        description="二维数组：defects[i] 对应 images[i] 的缺陷列表；该图无缺陷为 []",
    )


class PredictResponse(BaseModel):
    code: int = Field(..., description="0 成功；非 0 业务失败（常用 500）")
    msg: str = Field(..., description="ok 或错误说明")
    engine: str = Field("scanly-defects-v1", description="引擎标识")
    infer_type: str = Field("obb", description="本次实际使用的推理类型 detect / obb；缺省为 obb")
    board_id: str = Field("", description="板材 ID，原样回显")
    board_ok: bool = Field(..., description="全部图均无 ng=true 时为 true")
    laminating: bool = Field(False, description="是否按覆膜规则过滤")
    elapsed_ms: float = Field(0, description="本次耗时毫秒")
    results: list[CameraDefects] = Field(..., description="按入参相机顺序的缺陷分组")

    model_config = ConfigDict(json_schema_extra={"example": PREDICT_SUCCESS_EXAMPLE})


class ApiError(BaseModel):
    code: int = Field(500, description="业务错误码")
    msg: str = Field(..., description="错误说明")

    model_config = ConfigDict(json_schema_extra={"example": API_ERROR_EXAMPLE})


class SizeRule(BaseModel):
    height_mm: float = 0
    logic: str = Field("or", description="or / and / length_only，也可写 或/与/仅长度")
    length_mm: float = 0


class SchemeDefect(BaseModel):
    name: str
    cn_name: str = ""
    enable: bool = True
    confidence: float = 0.3
    detect_when_laminating: bool = False
    area_mm2: float = 0
    ng: SizeRule = Field(default_factory=SizeRule)
    label: SizeRule = Field(default_factory=SizeRule)
    discharge: SizeRule = Field(default_factory=SizeRule)


class SchemeChannels(BaseModel):
    model_config = ConfigDict(extra="allow")
    ng: bool = Field(True, description="精度页「NG」勾选")
    ng_area: bool = Field(True, description="NG 旁「面积」勾选；套用本行 area_mm2")
    label: bool = Field(True, description="精度页「贴标」勾选")
    label_area: bool = Field(False, description="贴标旁「面积」勾选")
    discharge: bool = Field(True, description="精度页「排出」勾选")
    discharge_area: bool = Field(False, description="排出旁「面积」勾选")


class SchemeView(BaseModel):
    code: int = 0
    msg: str = "ok"
    scheme_name: str = ""
    mm_per_px: float = 0.03
    channels: SchemeChannels = Field(default_factory=SchemeChannels)
    defects: list[SchemeDefect] = Field(default_factory=list)
    persisted: bool | None = Field(default=None, description="仅配置接口返回：是否已写盘")


class SchemeUpdate(BaseModel):
    model_config = ConfigDict(extra="allow")
    scheme_name: str | None = None
    mm_per_px: float | None = None
    channels: SchemeChannels | None = None
    defects: list[SchemeDefect] | None = None
    persist: bool = Field(True, description="是否写回 scheme.json")
    replace: bool = Field(False, description="true 时 defects 整表替换")


class LabelItem(BaseModel):
    name: str
    cn_name: str
    is_defect: bool
    in_train_set: bool


class LabelsResponse(BaseModel):
    code: int = 0
    engine: str = "scanly-defects-v1"
    items: list[LabelItem]


class HealthResponse(BaseModel):
    ready: bool
    model_loaded: bool
    infer_type: str = Field("obb", description="配置默认推理类型；缺省 obb")
    detect_loaded: bool = False
    obb_loaded: bool = False
    model_path: str = ""
    obb_model_path: str = ""
    scheme_name: str = ""
    msg: str = "ok"


PREDICT_HTTP_RESPONSES: dict[int | str, dict[str, Any]] = {
    200: {
        "description": "HTTP 始终 200。成功 code=0，body 为 PredictResponse；业务失败 code=500",
        "content": {
            "application/json": {
                "examples": {
                    "success": {
                        "summary": "推理成功（detect）",
                        "value": PREDICT_SUCCESS_EXAMPLE,
                    },
                    "success_obb": {
                        "summary": "推理成功（obb）",
                        "value": PREDICT_SUCCESS_OBB_EXAMPLE,
                    },
                    "error": {
                        "summary": "业务失败",
                        "value": API_ERROR_EXAMPLE,
                    },
                }
            }
        },
    }
}
