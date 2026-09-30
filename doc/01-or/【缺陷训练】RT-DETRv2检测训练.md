# 背景

封边缺陷检测训练入口目前是 Ultralytics YOLO11（`scanly/train/train.py`）。现场需要再增加一套 **RT-DETRv2 的 PyTorch 训练**，与 YOLO 共用同一批原始数据、同一套类别过滤与划分，便于对比检出效果。RT-DETRv2 读 COCO 标注，不能直接吃 YOLO `data.yaml`；训练代码必须落在 scanly 工程内、自包含，不得 import 其他工程目录里的实现。

# 目标

1. 在 `scanly/train/` 提供 RT-DETRv2（torch）检测训练入口，IDE 改变量即可运行。
2. 复用现有检测数据准备：原始目录、`TRAIN_CLASSES`、划分比例、空标签与拷贝/软链行为与 `train.py` 对齐。
3. 将已准备的 YOLO 检测目录转为 COCO json 后启动 RT-DETRv2 训练，写出 checkpoint。
4. 将 RT-DETRv2 基础实现复制到 scanly 新目录自包含复用，不引用仓库内其他工程的训练代码。
5. 训练正常结束后自动给出 val 指标；若已划分 test（`TEST_RATIO>0`）再给出 test 指标。

# 要求

## 功能性要求

1. **数据配置对齐 `train.py`**：入口暴露相同数据项：`SOURCE_DATA_ROOT` / `VOC_XML_PATH`、`TRAIN_CLASSES`、`OUTPUT_DIR`、`SKIP_PREPARE`、`PREPARE_ONLY`、`VAL_RATIO`、`TEST_RATIO`、`SEED`、`KEEP_EMPTY_LABELS`、`COPY_IMAGES`、`SINGLE_CLS`。数据准备调用现有 `prepare_dataset.py`（`label_format=detect`），写出 YOLO 检测目录后再转 COCO。
2. **类别**：`TRAIN_CLASSES` 顺序即 class id（从 0）。`SINGLE_CLS=True` 时不改写已预处理 YOLO 标签，仅在转 COCO 时把全部框视为一类。
3. **训练入口**：`scanly/train/train_rtdetrv2.py`。参数放在 `__main__` 变量区，不强制 argparse。可只准备数据、不训练。
4. **模型实现**：基础代码复制到 `scanly/train/rtdetrv2_pytorch/`（src、官方 configs、依赖声明）。训练通过该目录内 YAML + solver 启动。默认骨干 RT-DETRv2-L（ResNet-50），可用配置切换 R18 / R34；从 COCO 检测预训练微调，不再下载 ImageNet backbone。
5. **输出**：权重写到 `scanly/train/runs/` 下独立 run 目录（如 `rtdetrv2` / `rtdetrv2_2`…），不覆盖已有目录。默认 `EPOCHS=200`、`PATIENCE=25`（val 指标连续不升高则早停）。可选在训练结束后导出同目录 ONNX（本需求不要求接入现有 YOLO 加密导出）。
6. **验证**：训练入口提供只评估开关（`TEST_ONLY`，加载已有 checkpoint 做 **val**，不跑 test）。`fit` 正常结束后须自动再评估：先 val，若 `TEST_RATIO>0` 且 `images/test` 有图再 test。权重优先 `best.pth`，否则 `last.pth`。指标写入该次 run 目录。事后补评仍用独立入口，见 `scanly/doc/01-or/【缺陷训练】RT-DETRv2测试集验证.md`。无 checkpoint 时跳过训练后评估与 ONNX，并打警告。`TEST_RATIO>0` 但 test 无图时跳过 test、不失败。

## 非功能性要求

1. 不修改原始 `样本数据/` 与 VOC 源目录。
2. 不 import scanly 以外工程目录中的训练实现。
3. 本版不要求推理服务加载 RT-DETRv2 权重；不替代 YOLO11 训练入口。
4. RT-DETRv2 依赖（torch、faster-coco-eval 等）与 YOLO 依赖分开声明，避免绑死同一 `requirements.txt`。

# 当前工作项

# 已完成工作项

## 需求

- 新增 RT-DETRv2 torch 训练：复制自包含实现到 `scanly/train/rtdetrv2_pytorch/`，入口对齐 `train.py` 数据配置
- 训练结束后自动 val，再按 `TEST_RATIO>0` 做 test 评估

## 问题

（无）
