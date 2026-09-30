# 问题描述

在已配置 Go / ORT CUDA11（`LD_LIBRARY_PATH` 含 `cuda-11.8` / 系统 cuDNN）的训练机上运行：

```bash
python3 convert_2_onnx.py
```

`HALF=True`、`DEVICE=0` 时，Ultralytics 导出 ONNX 在 dry-run 前向（`Conv2d`）失败：

```text
RuntimeError: GET was unable to find an engine to execute this computation
```

日志可见权重已加载（如 YOLO11m-obb），失败发生在 `model.export(...)` 内部。

# 关联文档

| 类型 | 文档路径 |
|:---|:---|
| 原始需求 | `scanly/doc/01-or/【缺陷推理】ONNX导出加密与Go边端GPU授权.md` |
| 开发设计 | `scanly/doc/02-dr/缺陷训练/02.YOLO导出ONNX与加密.md` |
| 测试设计 | `scanly/doc/03-tr/缺陷训练/YOLO导出ONNX与加密测试设计.md` |
| 运维 | `scanly/doc/99-ops/train.md`、`scanly/doc/99-ops/run_go.md`、`scanly/doc/99-ops/run_go_cuda11.md` |

# 根因分析

1. 现场/训练机为 Go ORT 1.18.1 常设置 `LD_LIBRARY_PATH` 指向 **CUDA 11.8 + 系统 cuDNN8**（见 `run_go_cuda11.md`）。
2. Ultralytics 导出使用的 PyTorch 为 **自带 CUDA/cuDNN 的 wheel**（例：`torch-2.7.0+cu126`）。
3. 进程加载时优先使用 `LD_LIBRARY_PATH` 中的系统 cuDNN，与 PyTorch 二进制不匹配，GPU 卷积无法选到可用 engine，报 `GET was unable to find an engine...`。
4. 与 V100 是否支持 FP16 无关；FP32 导出或 `LD_LIBRARY_PATH=` 清空后再导出通常可通。

# 涉及文件

| 文件 | 变更 |
|:---|:---|
| `scanly/train/convert_2_onnx.py` | 导出前默认清空本进程 `LD_LIBRARY_PATH`；捕获该错误并提示 `HALF=False` / 手工 `LD_LIBRARY_PATH=` |
| `scanly/doc/02-dr/缺陷训练/02.YOLO导出ONNX与加密.md` | 补充环境约束 |
| `scanly/doc/03-tr/缺陷训练/YOLO导出ONNX与加密测试设计.md` | 增加回归用例 |
| `scanly/doc/99-ops/train.md` / `run_go.md` | 补充排障命令 |

# 设计约束更新

- GPU / FP16 导出时，**不得**让系统 CUDA11/cuDNN 路径覆盖 PyTorch wheel 自带库；脚本默认 `SANITIZE_LD_LIBRARY_PATH=True`。
- 临时规避：`LD_LIBRARY_PATH= python3 convert_2_onnx.py`，或 `HALF=False` 出 FP32（推理侧已兼容）。
