# 背景

板材封边缺陷检测的边端客户端（鲲鹏物联「木板封边缺陷检测系统」）已能配置缺陷中英文对照、启停以及按缺陷的面积 / NG / 贴标 / 排出距离与置信度。算法侧需要替换原 AI 推理步骤：边端把**本机已落盘图像路径**交给本服务，服务读盘推理后返回结构化缺陷列表与判定结果。

现有昆虫检测服务（`insect/script/predict_all.py` + Gradio HTTP）已验证「本机路径 + JSON HTTP + Gradio 调试」的对接方式。家装缺陷检测需沿用该调用形态，并接入封边缺陷类别与客户方案参数。第一版以**接口契约可对接**为优先，三路相机板尾汇总等能力可后续迭代。

客户输入：

- 标签与中文名：`scanly/doc/00-客户资料/业务相关/标签分类.txt`
- 缺陷启停与 AI 标签：`scanly/doc/00-客户资料/业务相关/缺陷配置.png`
- 判定参数：`scanly/doc/00-客户资料/业务相关/算法参数.png`

# 目标

1. 提供本机 HTTP JSON 推理接口，客户端传本地图像路径（边端主路径）。
2. 另提供 multipart 上传推理接口，便于跨机调试与无共享盘场景传图。
3. 按客户方案支持缺陷启停、中英文标签映射，以及面积 / NG / 贴标 / 排出距离 / 置信度 / 覆膜是否检测。
4. 返回每条缺陷的框、物理尺寸（mm）、置信度，以及 NG / 贴标 / 排出判定，供边端写 PLC 与 UI。
5. 交付一份可供客户端开发对照的接口文档（请求/响应/错误码/类别表）。
6. 保留 Gradio 页面，便于算法人员上传图片、调参、核对角框，而不影响 HTTP 契约。
7. 支持检测与 OBB 两类推理：配置默认类型，请求可指定 `infer_type`。

# 要求

## 功能性要求

1. **传输（路径）**：HTTP JSON。`POST /v1/predict` 入参为相机号 + 本机图片路径列表（每路相机可多张图）；服务按路径读盘。无图的相机关可省略，不要求一次齐套。**既有路径接口保持不变。**
2. **传输（上传）**：新增 `POST /v1/predict/upload`，`multipart/form-data` 上传图像二进制；元数据用表单字段（含可选 JSON `meta`）描述相机分组与 `infer_type` 等。响应结构与路径接口一致。实现落在 Python 服务（`scanly/defects/predict`）；Go 产线服务本迭代可不实现上传。
3. **类别**：AI 标签以模型输出英文名为准；中文显示名按客户缺陷管理表映射。训练标签与客户 UI 不完全重合时，以对照表同时收录，未启用类不参与判定。
4. **方案参数**：支持方案名称；每类可配置面积阈值，以及 NG / 贴标 / 排出各自的高度+逻辑（或/与/仅长度）+长度、置信度、覆膜时是否检测。精度页表头「NG / 贴标 / 排出」各有独立勾选，且**各自旁边有独立的「面积」勾选**（不是全局一个面积开关）。提供**查询与配置** HTTP 接口（与 `GET /v1/scheme` 同一套字段）；配置后立即作用于后续预测。预测请求仍可带 `scheme` 作单次覆盖。未在预测里下发时，用服务当前方案。
5. **判定**：先按该类置信度过滤，再按像素框换算 mm 后套用尺寸规则；逻辑为「或 / 与 / 仅长度」。某通道勾了「面积」且该类 `面积>0` 时，该通道还须框面积大于该阈值；面积阈值为 0 或未勾该通道「面积」则该通道不按面积过滤。
6. **调试**：同一进程提供 Gradio；页面改参仅本次内存生效。边端通过方案配置接口写入的参数须落盘（`scheme.json`），进程重启后仍生效。
7. **实现位置**：推理与服务代码放在 `scanly/defects/predict/`。
8. **文档**：形成客户端可独立阅读的 HTTP 接口文档；`/docs`（Swagger）须能看到请求/响应 JSON 结构（字段与示例），不得只显示 `"string"`。
9. **推理类型**：服务配置与预测请求均可选 `infer_type`：`detect`（轴对齐检测）或 `obb`（旋转框）。**缺省为 `obb`**。可分别配置检测权重与 OBB 权重；请求未带类型时用配置默认值（配置也未写则为 obb）。**推理输出统一按 OBB 结构**：含四点 `obb.points`；detect（bbox）结果视为 `angle=0` 的特殊 OBB（矩形四顶点）。轴对齐 `location` 仍保留便于边端兼容。
10. **模型目录**：权重默认放在 `scanly/defects/predict/models/`。`predict.json` 的 `model_path` / `obb_model_path` 写文件名或相对路径时，相对该目录解析（如 `"obb_model_path": "cam2-0819.pt"` → `models/cam2-0819.pt`）；绝对路径仍按原路径加载。
11. **ONNX 权重**：Python 服务须能加载明文 `.onnx`（不接 Go 加密链路）。支持两类：
    - **YOLO 导出**（`convert_2_onnx.py` / Ultralytics export）：经 Ultralytics 加载，`infer_type` 仍为 `detect` / `obb`。
    - **RT-DETRv2 导出**（`train_rtdetrv2.py` → `images` + `orig_target_sizes` → `labels/boxes/scores`）：仅轴对齐检测，对应 `infer_type=detect`；须可配 `imgsz`（与训练一致）与 `class_names`（缺省单类 `defect`）。
12. **配置可读性**：`predict.json` 须带字段备注（如 `_readme`），至少说明 `infer_type` 与 `model_path` / `obb_model_path` 如何配对，避免误配。
13. **目录批量离线推理**：提供本地脚本，对指定图像目录整批推理；输出目录含画框图与 Pascal VOC（`JPEGImages` + `Annotations`/*.xml，可选 `predefined_classes.txt`），便于 LabelImg 复核。入口用 `__main__` 变量配置路径，不强制 CLI。

## 非功能性要求

1. 内网本机调用，第一版不鉴权、不落库。
2. 业务错误采用 HTTP 200 + JSON 内错误码（与昆虫侧对接习惯一致），便于边端统一解析。
3. 模型未就绪时，健康检查与预测接口须明确失败原因，不得静默返回 OK。
4. 入口参数放在 `__main__` 变量区（IDE 可直接运行），不强制 argparse。
5. 第一版不接管相机采图；相机号由边端定义（如 `ch1`）。返回缺陷按「相机 → 与入参图片一一对应的二维列表」组织。

# 当前工作项

# 已完成工作项

## 需求

- 目录批量离线推理：画框图 + VOC XML（`predict_dir.py`）
- Python 推理支持明文 ONNX（YOLO + RT-DETRv2）；`predict.json` 补充 `infer_type` 等字段备注
- 整理缺陷推理 HTTP 对接需求，完成接口/判定设计与测试设计，并实现 `scanly/defects/predict` 第一版（本机路径 HTTP + Gradio）；客户端文档见 `scanly/doc/02-dr/缺陷推理/01.HTTP接口文档.md`
- 补充方案参数配置接口（`PUT`/`POST /v1/scheme`，默认可落盘），与查询字段一致
- 推理接口改为相机号 + 图片路径列表，返回按相机、按图对齐的缺陷二维列表
- Swagger `/docs` 声明请求/响应模型，成功示例可见 `results[].defects[][]` 结构
- 方案面积勾选按精度页拆成 `ng_area` / `label_area` / `discharge_area`，与 NG/贴标/排出通道独立
- 推理服务支持 OBB：`predict.json` 增加 `infer_type` / `obb_model_path`；`POST /v1/predict` 增加 `infer_type`；结果含 `obb.points` 四点，判定按旋转框宽高
- 模型相对路径默认解析到 `scanly/defects/predict/models/`；推理输出统一 OBB 结构，detect bbox 转为 angle=0 的四点框
- 新增 `POST /v1/predict/upload` multipart 上传推理（Python）；保留路径接口 `/v1/predict` 不动；Go 本迭代可不实现上传

## 问题

（无）
