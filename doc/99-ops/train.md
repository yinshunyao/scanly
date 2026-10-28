# 缺陷训练运行说明

实现入口：

- 检测训练：`scanly/train/train.py`
- 检测 core 训练：`scanly/train/train_core.py`（`fit` 结束后自动 val，`TEST_RATIO>0` 且 test 有图时再 test）
- 检测 core 单独导出 ONNX：`scanly/train/export_core.py`（`.pth` → 明文 `.onnx`；**不是** YOLO 加密链路）
- 检测测试集验证：`scanly/train/test.py`（YOLO）；`scanly/train/test_core.py`（检测 core `.pth` / `.onnx`，ONNX 强制 batch=1）
- OBB：`scanly/train/train_obb.py`
- 检测 mark 训练：`scanly/train/train_mark.py`（读公共数据配置；直接用准备好的 YOLO 目录）
- 检测 mark test 画框：`scanly/train/test_mark.py`（不训练；图写到权重所在 run 目录的 `vis/`）
- 导出加密 ONNX：`scanly/train/tools/convert_2_onnx.py`（Ultralytics YOLO `.pt`）

数据准备：`scanly/train/train_detect_cfg/prepare_dataset.py`（`label_format=detect|obb`）。设计：`scanly/doc/02-dr/缺陷训练/01.数据集准备与YOLO训练.md`、`scanly/doc/02-dr/缺陷训练/02.YOLO导出ONNX与加密.md`、`scanly/doc/02-dr/缺陷训练/05.检测core训练.md`、`scanly/doc/02-dr/缺陷训练/06.检测core测试集验证.md`。

已归档 run（v1.1–v2.1）的 val 曲线与结论：`scanly/doc/测试结果/各版本训练日志分析.md`（磁盘副本 `/Volumes/shunyao-h1/scanly/训练结果/说明.md`）。

## 依赖

建议与推理同一套 Ultralytics 环境：

```bash
pip install -r scanly/train/requirements.txt
```

预训练权重首次运行由 Ultralytics 自动下载：检测 `yolo11s.pt`，OBB `yolo11s-obb.pt`。

检测 core 另装 `scanly/train/detect_core/requirements.txt`（torch、faster-coco-eval、onnxruntime 等）。COCO 检测预训练按入口 `TUNING_URL` 首次下载到 `detect_core/pretrained/`。评估导出 ONNX 需要 `onnxruntime`（GPU 机可装 `onnxruntime-gpu` 以启用 CUDA EP）。

检测 mark 的运行库在 `scanly/train/requirements.txt` 中，并带上训练附加依赖（需要 Python 3.10 及以上）。只装基础包时，开训会缺训练依赖并停下。

## 配置

数据与训练配置已解耦（对齐 insect）：

| 路径 | 作用 |
|:---|:---|
| `scanly/train/train_detect_cfg/data_cfg.json` | 公共数据：源路径、`voc_xml_path`、`train_classes`、`val_classes`（core 选模 / mark 评估）、划分、`output_dir`（数据与训练同根） |
| `scanly/train/train_detect_yolo/train_config.json` | YOLO 检测超参（`train.py`）；可覆盖 `output_dir` |
| `scanly/train/train_detect_core/train_config.json` | 检测 core 超参 / `backbone`（自动关联 yaml）/ 预训练 / `run_prefix`（`train_core.py`） |
| `scanly/train/train_detect_obb/train_config.json` | OBB 超参（可覆盖公共数据批次与 `output_dir`；`train_obb.py`） |
| `scanly/train/train_detect_mark/train_config.json` | 检测 mark 规格 `size` 与超参（`train_mark.py`）；权重在 `output_dir/<run_prefix>` |

入口 `__main__` 仅改 `CONFIG_PATH`（及检测 core 的 `TEST_ONLY` / `RESUME`）。改数据或输出目录只动 `train_detect_cfg`；改超参只动对应入口 JSON。`voc_xml_path` 可为字符串或目录列表，非空优先于 `source_data_root`。多目录时 `voc_split.split_each_dir` 默认 `true`：每个目录单独按比例划分再合并；`false` 则全部打乱后一次划分。设计见 `scanly/doc/02-dr/缺陷训练/07.数据配置与训练配置解耦.md`。

`test.py` / `test_core.py` 仍可在变量区指定权重与划分；`output_dir` / imgsz / batch 宜与对应训练配置一致。

不修改原始 `样本数据/`。

## 启动

```bash
# 轴对齐检测（YOLO11）
python scanly/train/train.py

# 轴对齐检测（检测 core；数据配置与 train.py 对齐）
python scanly/train/train_core.py

# 轴对齐检测（检测 mark；数据配置与 train.py 对齐）
python scanly/train/train_mark.py

# 检测 mark 只做 test 画框（该次 run 的 vis/）
python scanly/train/test_mark.py

# 测试集验证（改 test.py 内 MODEL_PATH 指向 best.pt；需 data.yaml 含 test）
python scanly/train/test.py

# 检测 core 测试集验证（MODEL_PATH 指向 best.pth 或 best.onnx；MODEL_YML 与训练一致）
# .onnx 时 dataloader batch 会被强制为 1
python scanly/train/test_core.py

# 检测 core 单独导出 ONNX（改 export_core.py 内 CHECKPOINT / IMGSZ / SINGLE_CLS）
python scanly/train/export_core.py

# OBB 旋转框
python scanly/train/train_obb.py

# YOLO 训练权重转加密 ONNX（改 tools/convert_2_onnx.py；与 export_core 无关）
python scanly/train/tools/convert_2_onnx.py

# 若报 GET was unable to find an engine（常见：本机为 Go ORT 配了 CUDA11 的 LD_LIBRARY_PATH）
# 脚本默认会清空本进程 LD_LIBRARY_PATH；也可手工：
LD_LIBRARY_PATH= python3 scanly/train/tools/convert_2_onnx.py
# 或改 HALF=False 导出 FP32（推理侧已兼容）
```

或 IDE 打开对应脚本运行。YOLO 权重默认写到 `scanly/train/runs/train/`（已存在则 `train2`、`train3`…）。检测 core 写到 `scanly/train/runs/detect_core/`（已存在则 `detect_core_2`…）。训练结束后该 run 目录含 `val_metrics.json`；`TEST_RATIO>0` 且有 test 图时还有 `test_metrics.json`。YOLO `test.py` 指标默认 `runs/<RUN_NAME>/test_metrics.json`；检测 core `test_core.py` 默认 `runs/detect_core-test/test_metrics.json`。`test_core.py` 默认 `SAVE_VIS=True`：同目录写出 `vis/` 画框图（TP 绿 / FP 红 / FN 橙，预测旁置信度）、`predictions.jsonl`（含 `score`）、`vis_summary.json`；`VIS_CONF` / `VIS_IOU` 只影响画图与记录，不影响 mAP。检测 core 训练结束的 test 评估默认把画框图写到该次 run 目录的 `vis/`（`save_vis` 可关），并在 `test_metrics.json` 写入推理时延；val 不画图、不计时延。检测 mark 的 test 画框写在该次 run 目录的 `vis/`，匹配统计写在该目录的 `test_summary.json`（与 `vis/vis_summary.json` 相同，含按类 TP/FP/FN 和至少中一条的图片数）并含推理时延，`predictions.jsonl` 含 IoU；运行库能评估 test 时同目录还有 `test_metrics.json`。检测 / OBB 的 `best.pt` 可配到 Python 推理 `predict.json`，或经 `tools/convert_2_onnx.py` 加密后交给 `scanly/defects/predict_go`（`HALF=True` 导出 FP16，推理侧已兼容 float16 I/O）。本版检测 core 的 `.pth` / `.onnx` 不接入现有加密推理。
