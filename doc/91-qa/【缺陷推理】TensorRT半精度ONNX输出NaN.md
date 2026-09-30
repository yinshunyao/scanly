# 问题描述

V100 上 `DEVICE=tensorrt` 跑 `test_onnx_infer_perf.py`：

- FP32 密文 + TensorRT EP（`trt_fp16=True`）成功：`session providers` 首位为 TensorRT，`session.run` 约 6.6ms。
- 随后加载 FP16 密文（已是 `tensor(float16)`）再建 TensorRT 会话，warmup/`predict_rgb` 在 `xywhr_to_points` 抛错：`ValueError: cannot convert float NaN to integer`。

INT64→INT32、FP16 subnormal 权重警告本身不是失败。

# 关联文档

| 类型 | 文档路径 |
|:---|:---|
| 原始需求 | `scanly/doc/01-or/【缺陷推理】ONNX导出加密与Go边端GPU授权.md` |
| 开发设计 | `scanly/doc/02-dr/缺陷推理/03.Go端ONNX加密推理与GPU授权.md` |
| 测试设计 | `scanly/doc/03-tr/缺陷推理/Go端ONNX加密推理与GPU授权测试设计.md` |

# 根因分析

TensorRT 推荐路径是 **FP32 ONNX + EP 内 `trt_fp16_enable`**（在引擎里做 FP16 tactic）。  
把 **已经是 FP16 的 ONNX** 再交给 TensorRT 且仍开 `trt_fp16_enable`，等于半精度上再叠半精度，V100 / TRT 8.6 上 OBB 头易出 NaN/Inf，解码 `round` 转 int 崩溃。

这与是否重导 ONNX 无关；FP32+TRT 已证明现网密文可用。

# 涉及文件

- `defects/predict_go/server_onnx.py`：已是 FP16 的图关闭额外 TRT FP16；解码跳过非有限框
- `defects/predict_go/internal/infer/engine.go`：解码同样跳过 NaN/Inf
- `defects/test/test_onnx_infer_perf.py`：`DEVICE=tensorrt` 时默认不测 FP16 ONNX
- `doc/99-ops/run_go_cuda11.md`：写明 TensorRT 用 FP32 密文

# 设计约束更新

`device=tensorrt` 的半精度加速以 **TRT EP FP16 + FP32 ONNX** 为准。FP16 ONNX 仍给 `device=cuda` 用；不要与 TRT FP16 叠用。
