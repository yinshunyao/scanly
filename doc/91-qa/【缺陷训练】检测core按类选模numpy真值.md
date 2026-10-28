# 问题描述

检测 core 训练开了 `val_classes`（日志为 `GlueSeam,Tackless`）。第 1 个 epoch 训练和 Val 进度条都走完后崩溃：

```
ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
```

堆栈在 `class_scores_from_coco_eval`：`getattr(params, "iouThrs", []) or []`。

同一轮训练日志还有 `det_engine.py` 警告：把 `requires_grad=True` 的 loss 张量交给 `math.isfinite`。

# 关联文档

| 类型 | 文档路径 |
|:---|:---|
| 原始需求 | `scanly/doc/01-or/【缺陷训练】检测core训练.md` |
| 开发设计 | `scanly/doc/02-dr/缺陷训练/05.检测core训练.md` |
| 测试设计 | `scanly/doc/03-tr/缺陷训练/检测core训练测试设计.md` |

# 根因分析

faster-coco-eval 的 `params.iouThrs` 是 `np.linspace` 得到的多元素数组。`array or []` 会求数组的真值，从而抛出上述 ValueError。评估本身已完成，失败发生在按 `val_classes` 汇总 fitness、决定是否写 `best.pth` 之前。`catIds` 多为 list，所以先通过，崩在下一行的 `iouThrs`。

`val_classes` 为空时不走这条路径，所以只有指定选模类别才会在首轮 val 后暴露。

loss 警告是另一处：`reduce_dict` 之后的张量仍挂着梯度，`math.isfinite` 把它收成 Python 标量时触发警告。训练不会因此停下。

# 涉及文件

- `scanly/train/detect_core/src/solver/det_solver.py`：`catIds` / `iouThrs` / `stats` 用 `_as_seq` 转成 list，不再 `or []`。
- `scanly/train/detect_core/src/solver/det_engine.py`：汇总 loss 前 `detach`，再判断是否有限。

# 设计约束更新

`05.检测core训练.md`：按类选模读取 `precision` / `recall` / `catIds` / `iouThrs` 时按序列转换，不得对 numpy 数组做真值判断。
