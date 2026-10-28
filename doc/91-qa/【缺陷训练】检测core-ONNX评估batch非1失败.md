# 问题描述

`train_core.py` 训后导出 ONNX 成功，但对 test 做 ONNX 评估时失败：

```text
ONNX Runtime ... Reshape node. Name:'node_view'
Input shape:{4,256,32,32}, requested shape:{1,256,1024}
```

复现条件：`export_onnx=true`、训练 `batch=4`、`imgsz=1024`，fit 结束后自动用刚导出的 `best.onnx` 跑 `test_core.run_val`（dataloader batch 与训练同为 4）。

# 关联文档

| 类型 | 文档路径 |
|:---|:---|
| 原始需求 | `scanly/doc/01-or/【缺陷训练】检测core训练.md`、`scanly/doc/01-or/【缺陷训练】检测core测试集验证.md` |
| 开发设计 | `scanly/doc/02-dr/缺陷训练/05.检测core训练.md`、`scanly/doc/02-dr/缺陷训练/06.检测core测试集验证.md` |
| 测试设计 | `scanly/doc/03-tr/缺陷训练/检测core训练测试设计.md`、`scanly/doc/03-tr/缺陷训练/检测core测试集验证测试设计.md` |
| 运维 | `scanly/doc/99-ops/train.md` |

# 根因分析

1. `detect_core/src/misc/onnx_export.py` 用 **batch=1** 的 dummy 做 `torch.onnx.export`，并声明 `dynamic_axes` 的 batch 维为 `N`。
2. HybridEncoder 内 `nn.MultiheadAttention` / 若干 `view`/`Reshape` 在经典导出器下会把 **导出时的 batch=1** 写进图常量（如 `{1,256,1024}`，对应 `C=256`、`H*W=32*32`）。
3. 评估侧仍用训练 `batch=4` 喂 ORT → 输入 `{4,256,32,32}` 无法 reshape 到 `{1,256,1024}`。
4. 与「导出本身失败」不同：`.onnx` 文件已写出；失败发生在 **ONNX 推理 batch≠1**。上游 deploy 样例也始终 batch=1。

# 涉及文件

| 文件 | 变更 |
|:---|:---|
| `scanly/train/test_core.py` | ONNX 评估强制 `batch=1`（`resolve_onnx_eval_batch`） |
| `scanly/train/detect_core/src/misc/onnx_export.py` | 文档说明；`dynamo=False` 避免新 exporter 假降级 |
| `scanly/train/export_core.py` | 新增：单独 `.pth`→`.onnx`（非 YOLO `convert_2_onnx`） |
| `scanly/doc/02-dr/缺陷训练/05/06`、`03-tr`、`99-ops/train.md` | 同步约束与入口 |

# 设计约束更新

- 检测 core 明文 ONNX **有效 batch 固定为 1**；`test_core` / 训后 test 遇 `.onnx` 自动改 batch=1。
- 单独导出用 `export_core.py`；YOLO 加密链路仍用 `tools/convert_2_onnx.py`，二者不可混用。
