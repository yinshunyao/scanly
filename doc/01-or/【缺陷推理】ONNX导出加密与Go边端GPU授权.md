# 背景

边端推理现以 Python + Ultralytics `.pt` 交付，源码与权重在现场容易被复制。需要把训练产物导出为 ONNX、由 Python 加密后交付，再用 Go 服务解密推理，并按现场 NVIDIA GPU UUID 签发许可证，避免换机直接使用。

Python 调试页（Gradio）保留在原 `defects/predict/`，本需求的 Go 服务面向产线，不提供 Gradio。

# 目标

1. 在 `scanly/train/` 提供与训练入口配套的 ONNX 导出脚本，可对 YOLO 检测 / OBB 权重导出（含可选 FP16 半精度）并加密。
2. 在 `scanly/defects/predict_go/` 用 Go 加载加密 ONNX，提供与现网一致的 HTTP 推理契约（无 Gradio）。
3. 用 Python 签发 License：每份可配置 GPU 列表；每张卡绑定 uuid、name、serial 三项，Go 校验签名、有效期与本机 GPU 全字段后才解密模型。
4. 提供现场可独立阅读的帮助与运行说明（导出、加密、签发、启动、接口）。

# 要求

## 功能性要求

1. **导出**：`scanly/train/convert_2_onnx.py` 与 `train.py` / `train_obb.py` 同目录、同样用 `__main__` 变量入口（不强制 argparse）。可指定 `.pt` 路径、`detect`/`obb`、`imgsz`。导出后写出类别元数据。
2. **半精度导出**：支持 Ultralytics `half=True` 导出 FP16 ONNX，用于 GPU 推理加速；元数据须标明精度（`fp16` / `fp32`）。半精度导出须在 GPU 设备上执行（Ultralytics 约束）。Go / Python 推理须按模型输入输出 dtype 喂数与解码，兼容既有 FP32 与 FP16 密文。
3. **推理后端切换**：`predict.json` 的 `device` 支持 `cpu` / `cuda` / `tensorrt`（别名 `trt`、`trt_fp16`）。`tensorrt` 时启用 ONNX Runtime **TensorRT EP**，默认打开 FP16（`trt_fp16_enable`），并支持引擎缓存目录 `trt_engine_cache_path`；要求会话实际使用 TensorRT（失败不得静默当已加速成功）。Python 冒烟与 Go 产线均须支持该切换。
4. **性能对比脚本**：在 `scanly/defects/test/` 提供 Python 脚本，可对 FP32 / FP16 加密（或明文）ONNX 做同图 warmup + 计时，输出延迟统计与相对加速比（`__main__` 变量入口，不强制 argparse）；`DEVICE` 可切到 `tensorrt`。
5. **加密**：Python 用 AES-256-GCM 加密 ONNX；密钥单独保存（仅厂商侧）。交付现场的是密文，不是明文 `.onnx` / `.pt`。
6. **解密**：Go 仅在 License 校验通过后，用 License 内携带的密钥在内存解密，不把明文 ONNX 写盘。
7. **License**：Python 生成。字段至少包含 GPU 列表、签发时间、可选过期时间、模型标识、模型密钥。每张授权 GPU 必须同时绑定 `uuid`、`name`、`serial`（与 `nvidia-smi --query-gpu=uuid,name,serial --format=csv,noheader` 一致）。用厂商私钥签名；Go 内嵌公钥验签。本机任一 GPU 的三项全部与列表中某一条一致才视为通过；只匹配 UUID、型号或序列号之一不算通过。
8. **GPU 采集**：签发工具可读取本机上述 nvidia-smi 三列，便于现场登记；三项均须可手工填写（多卡、多机）。
9. **HTTP**：Go 服务实现现网预测 / 方案 / 标签 / 健康检查路径，不挂 Gradio、不要求 Swagger。另提供 License 状态查询。判定规则与 `scheme.json` 一致。
10. **帮助文档**：说明目录职责、导出加密、签发 License、依赖（ONNX Runtime 动态库）、启动与接口、`device=tensorrt` 与引擎缓存。

## 非功能性要求

1. 私钥与 AES 密钥不得随现场安装包分发。
2. License 无效、过期或 GPU 不匹配时，不得解密模型，预测接口明确失败原因。
3. 第一版不要求对抗专业逆向（内存 dump）；目标是提高现场拷贝门槛。
4. 无 NVIDIA GPU 的开发机允许服务启动，但不得判定 License 通过（除非签发工具与校验使用同一套真实 GPU uuid/name/serial）。
5. 本迭代不做 INT8 量化；半精度路径为 Ultralytics ONNX `half` 与/或 TensorRT FP16（`device=tensorrt`）。
6. TensorRT 首次建引擎可能较慢，须启用磁盘缓存以便后续启动复用。

# 当前工作项

# 已完成工作项

## 需求

- 训练目录新增 `convert_2_onnx.py`：YOLO `.pt` 转 ONNX 并 AES 加密
- 新增 `scanly/defects/predict_go`：Go 解密 ONNX、HTTP 推理（无 Gradio）、GPU UUID License 校验
- Python 签发 License；帮助文档与运行说明
- License 按 GPU uuid + name + serial 三项全绑定校验
- ONNX 半精度（FP16）导出：`convert_2_onnx.py` 的 `HALF`；meta `precision`；Go / Python 兼容 float16 I/O
- FP32/FP16 ONNX 推理性能对比脚本：`scanly/defects/test/test_onnx_infer_perf.py`
- `device=tensorrt`：Python / Go 启用 TensorRT EP + FP16，配置与性能脚本对齐

## 问题

- 现场 V100：ORT≥1.20/1.23 + cuDNN9 失败；Python **1.18.1** GPU 已通。Go 发布线改为 `scanly-defects-go-ort118`（API 18）+ ORT **1.18.1** CUDA11 包；手册 `doc/99-ops/run_go_cuda11.md`
- FP16 导出 `GET was unable to find an engine`：系统 `LD_LIBRARY_PATH`（CUDA11/cuDNN）与 PyTorch wheel 冲突；`convert_2_onnx.py` 默认清空本进程路径，见 `doc/91-qa/【缺陷训练】ONNX半精度导出GET引擎错误.md`
- V100 TensorRT：FP32 密文 + TRT FP16 可用；**FP16 ONNX 再开 TRT FP16** 解码 NaN。约束：TensorRT 用 FP32 密文，半精度走 EP 的 `trt_fp16_enable`。见 `doc/91-qa/【缺陷推理】TensorRT半精度ONNX输出NaN.md`
