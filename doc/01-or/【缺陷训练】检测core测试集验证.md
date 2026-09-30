# 背景

YOLO11 检测有独立入口 `scanly/train/test.py`，对已划分的 test 集做 mAP。RT-DETRv2 训练入口只有 `TEST_ONLY`（加载 checkpoint 做 **val**），没有对齐 `test.py` 的测试集评估；`test.py` 走 Ultralytics，读不了 RT-DETRv2 的 `.pth` / 导出 `.onnx`。对比 YOLO 与 RT-DETRv2 需要同一块测试集上的指标。

# 目标

1. 在 `scanly/train/` 提供 RT-DETRv2 测试集验证入口，IDE 改变量即可运行。
2. 对训练产出的 `.pth` 与导出 `.onnx` 在指定划分（默认 test）上计算 COCO mAP。
3. 写出与 YOLO `test.py` 同风格的 `test_metrics.json`（含 mAP50、mAP50-95、每类 AP）。

# 要求

## 功能性要求

1. **入口**：`scanly/train/test_core.py`。参数放在 `__main__` 变量区，不强制 argparse。
2. **数据**：复用已预处理的 YOLO 检测目录（`OUTPUT_DIR`），与 `train_core.py` 相同的 COCO 转换与 `SINGLE_CLS` 语义。默认评估 `SPLIT=test`；可改为 `val`。无对应划分或图像为空时报错，提示 `TEST_RATIO>0` 后重新准备。
3. **权重**：`MODEL_PATH` 为训练产出的 `.pth`（如 `best.pth`）或同目录导出的 `.onnx`。相对路径相对 `scanly/train/`。文件不存在则报错。YOLO `.pt` 须提示改用 `test.py`。
4. **架构配置**：`.pth` 评估须指定与训练一致的 `MODEL_YML`、`IMGSZ`、`SINGLE_CLS`。`.onnx` 预处理与训练 val 一致（`imgsz` 方形 resize、像素 /255）。
5. **指标**：COCO bbox AP（AP、AP50、AP75、AR100）及每类 AP50 / AP50-95。日志打印 mAP50 / mAP50-95；结果写入 `{TRAIN_PROJECT}/{RUN_NAME}/test_metrics.json`。默认 `RUN_NAME` 带 `-test` 后缀，不覆盖训练 run。
6. **不准备数据**：不对原始 VOC / YOLO 源再 prepare；只读 `OUTPUT_DIR`。

## 非功能性要求

1. 不 import scanly 以外工程目录中的训练实现。
2. 不改推理服务、不走 YOLO 加密导出链路。
3. 本版不要求可视化 plots；不替代 `train_core.py` 的 `TEST_ONLY` val。训练 `fit` 结束后的自动 val/test 见检测训练 OR，本入口负责事后补评。

# 当前工作项

# 已完成工作项

## 需求

- 新增 `test_core.py`：对 RT-DETRv2 `.pth` / `.onnx` 做测试集 COCO 评估

## 问题

（无）
