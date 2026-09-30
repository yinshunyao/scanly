# 缺陷推理服务运行说明

实现入口：`scanly/defects/predict/serve.py`。接口契约：`scanly/doc/02-dr/缺陷推理/01.HTTP接口文档.md`。

## 依赖

建议在已有 YOLO/Ultralytics 环境中安装：

```bash
pip install -r scanly/defects/predict/requirements.txt
```

## 配置

| 文件 | 作用 |
|:---|:---|
| `scanly/defects/predict/config/predict.json` | 权重、`infer_type`、切片、`mm_per_px`、端口；字段说明见同文件 `_readme` |
| `scanly/defects/predict/models/` | 默认模型目录；配置写文件名如 `cam2-0819.pt` 即加载此目录下文件 |
| `scanly/defects/predict/config/scheme.json` | 默认方案；也可由 `PUT/POST /v1/scheme` 覆盖写回 |

### `infer_type` 怎么配

| 目标 | `infer_type` | 权重 |
|:---|:---|:---|
| 轴对齐检测（YOLO / RT-DETRv2） | `detect` | `model_path` → `.pt` 或明文 `.onnx` |
| 旋转框（仅 YOLO-OBB） | `obb`（也可省略，默认） | `obb_model_path`；或只配 `model_path` 且留空 `obb_model_path` |

`infer_type` 选的是**任务类型**，不是文件后缀。RT-DETRv2 训练导出的 `.onnx`（输入 `images`+`orig_target_sizes`）只能 `detect`，并配 `imgsz`（与训练一致，常见 `1024`）、`class_names`（单类默认 `["defect"]`）。YOLO 经 Ultralytics/`convert_2_onnx` 导出的 `.onnx` 仍按 detect/obb 走 Ultralytics。加密 `.onnx.enc` 仍用 Go 服务，本 Python 入口不解密。

`model_path` / `obb_model_path` 都可空：服务能起来，但对应类型的预测与 `/health/ready` 会显示未就绪。相对路径相对 `models/` 解析；绝对路径原样。

也可在 `serve.py` 的 `__main__` 变量 `MODEL_PATH` / `OBB_MODEL_PATH` / `INFER_TYPE` / `SERVER_PORT` 覆盖（IDE 直接运行）。

预测请求对象可带 `infer_type` 覆盖本次类型；不传则用服务默认。需要检测时显式传：

```json
{"infer_type":"detect","cameras":[{"camera":"ch1","images":["/data/a.bmp"]}]}
```

缺陷结果统一带 `obb.points`；detect 时为矩形四顶点、`angle=0`。

## 目录批量离线（画框 + VOC）

入口：`scanly/defects/predict/predict_dir.py`。改 `__main__` 里 `INPUT_DIR` / `OUTPUT_DIR`（默认输入 `cam2-test`）后 IDE 运行。输出：

| 路径 | 内容 |
|:---|:---|
| `JPEGImages/` | 原图硬链或复制 |
| `Annotations/*.xml` | Pascal VOC |
| `vis/` | 画框图 |
| `predefined_classes.txt` / `summary.json` | 类名与汇总 |

试跑可设 `LIMIT=3`。依赖与权重与 `serve.py` 相同（读同一份 `predict.json`）。

## 启动

在 IDE 打开 `serve.py` 后运行，或：

```bash
python scanly/defects/predict/serve.py
```

默认 `0.0.0.0:37870`。

| 地址 | 用途 |
|:---|:---|
| `http://<host>:37870/` | Gradio 调试 |
| `http://<host>:37870/v1/predict` | 边端推理（本机路径 JSON） |
| `http://<host>:37870/v1/predict/upload` | 跨机/调试：multipart 上传图像 |
| `http://<host>:37870/docs` | Swagger |
| `http://<host>:37870/health/ready` | 就绪 |

日志走 stdout/stderr。后台建议：`nohup python serve.py > web.log 2>&1 &`。

## 与边端同机约定

- 边端采图落盘后，把**本机绝对路径**放入 JSON（产线主路径）。
- 跨机或无共享盘可用 `/v1/predict/upload` 直接传图。
- 路径模式下服务必须与采图进程看到同一文件系统路径（Windows 盘符、Linux 挂载点一致）。
- 第一版不鉴权，仅内网监听。
