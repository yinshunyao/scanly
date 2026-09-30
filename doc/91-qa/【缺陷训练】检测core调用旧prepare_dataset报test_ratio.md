# 问题描述

客户机运行 `python3 train_core.py` 立即退出：

```
TypeError: prepare_dataset() got an unexpected keyword argument 'test_ratio'
```

`train_core.py` 与较旧的 `prepare_dataset.py` 同目录（无 `test_ratio` 形参）。

# 关联文档

| 类型 | 文档路径 |
|:---|:---|
| 原始需求 | `scanly/doc/01-or/【缺陷训练】检测core训练.md` |
| 开发设计 | `scanly/doc/02-dr/缺陷训练/05.检测core训练.md` |
| 测试设计 | `scanly/doc/03-tr/缺陷训练/检测core训练测试设计.md` |
| 数据准备 | `scanly/doc/02-dr/缺陷训练/01.数据集准备与YOLO训练.md` |

# 根因分析

入口按当前仓库的 `prepare_dataset(..., test_ratio=...)` 调用。现场只同步了 `train_core.py`，`prepare_dataset.py` 仍是无 `test_ratio` 的版本，关键字参数被拒绝。

# 涉及文件

- `scanly/train/train_core.py`：按 `prepare_dataset` 实际签名过滤参数；不支持的项打警告后忽略。

# 设计约束更新

调用数据准备时不得假设现场 `prepare_dataset.py` 与入口同一提交；未知关键字须过滤。无 `test_ratio` 时只划分 train/val，不影响 RT-DETRv2 训练。
