# 缺陷训练运行说明

实现入口：

- 检测训练：`scanly/train/train.py`
- 检测 core 训练：`scanly/train/train_core.py`（`fit` 结束后自动 val，`TEST_RATIO>0` 且 test 有图时再 test）
- 检测测试集验证：`scanly/train/test.py`（YOLO）；`scanly/train/test_core.py`（检测 core `.pth` / `.onnx`）
- OBB：`scanly/train/train_obb.py`
- 导出加密 ONNX：`scanly/train/convert_2_onnx.py`（YOLO `.pt`）

数据准备：`scanly/train/train_detect_cfg/prepare_dataset.py`（`label_format=detect|obb`）。设计：`scanly/doc/02-dr/缺陷训练/01.数据集准备与YOLO训练.md`、`scanly/doc/02-dr/缺陷训练/02.YOLO导出ONNX与加密.md`、`scanly/doc/02-dr/缺陷训练/05.检测core训练.md`、`scanly/doc/02-dr/缺陷训练/06.检测core测试集验证.md`。

已归档 run（v1.1–v2.1）的 val 曲线与结论：`scanly/doc/测试结果/各版本训练日志分析.md`（磁盘副本 `/Volumes/shunyao-h1/scanly/训练结果/说明.md`）。

## 依赖

建议与推理同一套 Ultralytics 环境：

```bash
pip install -r scanly/train/requirements.txt
```

预训练权重首次运行由 Ultralytics 自动下载：检测 `yolo11s.pt`，OBB `yolo11s-obb.pt`。

检测 core 另装 `scanly/train/detect_core/requirements.txt`（torch、faster-coco-eval、onnxruntime 等）。COCO 检测预训练按入口 `TUNING_URL` 首次下载到 `detect_core/pretrained/`。评估导出 ONNX 需要 `onnxruntime`（GPU 机可装 `onnxruntime-gpu` 以启用 CUDA EP）。

## 配置

数据与训练配置已解耦（对齐 insect）：

| 路径 | 作用 |
|:---|:---|
| `scanly/train/train_detect_cfg/data_cfg.json` | 公共数据：源路径、`voc_xml_path`、`train_classes`、划分、`output_dir`（数据与训练同根） |
| `scanly/train/train_detect_yolo/train_config.json` | YOLO 检测超参（`train.py`）；可覆盖 `output_dir` |
| `scanly/train/train_detect_core/train_config.json` | 检测 core 超参 / yaml / 预训练 / `run_prefix`（`train_core.py`） |
| `scanly/train/train_detect_obb/train_config.json` | OBB 超参（可覆盖公共数据批次与 `output_dir`；`train_obb.py`） |

入口 `__main__` 仅改 `CONFIG_PATH`（及检测 core 的 `TEST_ONLY` / `RESUME`）。改数据或输出目录只动 `train_detect_cfg`；改超参只动对应入口 JSON。`voc_xml_path` 非空优先于 `source_data_root`。设计见 `scanly/doc/02-dr/缺陷训练/07.数据配置与训练配置解耦.md`。

`test.py` / `test_core.py` 仍可在变量区指定权重与划分；`output_dir` / imgsz / batch 宜与对应训练配置一致。

不修改原始 `样本数据/`。

## 启动

```bash
# 轴对齐检测（YOLO11）
python scanly/train/train.py

# 轴对齐检测（检测 core；数据配置与 train.py 对齐）
python scanly/train/train_core.py

# 测试集验证（改 test.py 内 MODEL_PATH 指向 best.pt；需 data.yaml 含 test）
python scanly/train/test.py

# 检测 core 测试集验证（MODEL_PATH 指向 best.pth 或 best.onnx；MODEL_YML 与训练一致）
python scanly/train/test_core.py

# OBB 旋转框
python scanly/train/train_obb.py

# 训练权重转加密 ONNX（改 convert_2_onnx.py 内 MODEL_PATH / HALF / DEVICE 后运行；HALF 默认 True 需 GPU）
python scanly/train/convert_2_onnx.py

# 若报 GET was unable to find an engine（常见：本机为 Go ORT 配了 CUDA11 的 LD_LIBRARY_PATH）
# 脚本默认会清空本进程 LD_LIBRARY_PATH；也可手工：
LD_LIBRARY_PATH= python3 scanly/train/convert_2_onnx.py
# 或改 HALF=False 导出 FP32（推理侧已兼容）
```

或 IDE 打开对应脚本运行。YOLO 权重默认写到 `scanly/train/runs/train/`（已存在则 `train2`、`train3`…）。检测 core 写到 `scanly/train/runs/detect_core/`（已存在则 `detect_core_2`…）。训练结束后该 run 目录含 `val_metrics.json`；`TEST_RATIO>0` 且有 test 图时还有 `test_metrics.json`。YOLO `test.py` 指标默认 `runs/<RUN_NAME>/test_metrics.json`；检测 core `test_core.py` 默认 `runs/detect_core-test/test_metrics.json`。检测 / OBB 的 `best.pt` 可配到 Python 推理 `predict.json`，或经 `convert_2_onnx.py` 加密后交给 `scanly/defects/predict_go`（`HALF=True` 导出 FP16，推理侧已兼容 float16 I/O）。本版检测 core 的 `.pth` / `.onnx` 不接入现有加密推理。
