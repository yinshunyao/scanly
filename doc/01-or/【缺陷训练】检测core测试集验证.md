# 背景

YOLO11 检测有独立入口 `scanly/train/test.py`，对已划分的 test 集做 mAP。RT-DETRv2 训练入口只有 `TEST_ONLY`（加载 checkpoint 做 **val**），没有对齐 `test.py` 的测试集评估；`test.py` 走 Ultralytics，读不了 RT-DETRv2 的 `.pth` / 导出 `.onnx`。对比 YOLO 与 RT-DETRv2 需要同一块测试集上的指标。

# 目标

1. 在 `scanly/train/` 提供 RT-DETRv2 测试集验证入口，IDE 改变量即可运行。
2. 对训练产出的 `.pth` 与导出 `.onnx` 在指定划分（默认 test）上计算 COCO mAP。
3. 写出与 YOLO `test.py` 同风格的 `test_metrics.json`（含 mAP50、mAP50-95、每类 AP 与 IoU=0.5 召回）。

# 要求

## 功能性要求

1. **入口**：`scanly/train/test_core.py`。参数放在 `__main__` 变量区，不强制 argparse。
2. **数据**：复用已预处理的 YOLO 检测目录（`OUTPUT_DIR`），与 `train_core.py` 相同的 COCO 转换与 `SINGLE_CLS` 语义。默认评估 `SPLIT=test`；可改为 `val`。无对应划分或图像为空时报错，提示 `TEST_RATIO>0` 后重新准备。
3. **权重**：`MODEL_PATH` 为训练产出的 `.pth`（如 `best.pth`）或同目录导出的 `.onnx`。相对路径相对 `scanly/train/`。文件不存在则报错。YOLO `.pt` 须提示改用 `test.py`。
4. **架构配置**：`.pth` 评估须指定与训练一致的 `MODEL_YML`、`IMGSZ`、`SINGLE_CLS`。`.onnx` 预处理与训练 val 一致（`imgsz` 方形 resize、像素 /255）。
5. **指标**：COCO bbox AP（AP、AP50、AP75、AR100）、每类 AP50 / AP50-95，以及每类在 IoU=0.5、每图最多 100 框时的召回。日志打印总体 mAP50 / mAP50-95，并按类各打一行 AP50、AP、召回。结果写入 `{TRAIN_PROJECT}/{RUN_NAME}/test_metrics.json`。默认 `RUN_NAME` 带 `-test` 后缀，不覆盖训练 run。`SINGLE_CLS=True` 时只有一类，按类统计退化为 `defect`。
6. **不准备数据**：不对原始 VOC / YOLO 源再 prepare；只读 `OUTPUT_DIR`。
7. **评估可视化（开关）**：`SAVE_VIS` 打开时，在评估过程中对每张图画预测框与 GT，标注正确与否（TP/FP/FN）与预测置信度；预测框置信度写入结果文件。关闭时行为与原先一致（只写 mAP）。画框与匹配逻辑须自包含于 `scanly/train/`，不得依赖 `defects/` 等 train 以外目录。

## 非功能性要求

1. 不 import scanly 以外工程目录中的训练实现；可视化不得引用 `defects/`。
2. 不改推理服务、不走 YOLO 加密导出链路。
3. 可视化仅影响落盘画图与预测记录，不得改变 COCO mAP 计算（评估路径仍不按分数阈值过滤 query）。不替代 `train_core.py` 的 `TEST_ONLY` val。训练 `fit` 结束后的自动 val/test 见检测训练 OR，本入口负责事后补评。

# 当前工作项

# 已完成工作项

## 需求

- 新增 `test_core.py`：对 RT-DETRv2 `.pth` / `.onnx` 做测试集 COCO 评估
- 评估可视化开关：画框图（TP/FP/FN + 置信度）并记录预测分数
- 指标按类写出 AP50、AP50-95 与 IoU=0.5 召回，并在日志中按类打印

## 问题

（无）
