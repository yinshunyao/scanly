# 问题描述

`test_core.py` 已跑完 val（日志有 AP / AP50），写出 `test_metrics.json` 前崩溃：

```
ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
```

堆栈在 `metrics_from_coco_eval`：`getattr(coco_eval, "stats", []) or []`。

# 关联文档

| 类型 | 文档路径 |
|:---|:---|
| 原始需求 | `scanly/doc/01-or/【缺陷训练】检测core测试集验证.md` |
| 开发设计 | `scanly/doc/02-dr/缺陷训练/06.检测core测试集验证.md` |
| 测试设计 | `scanly/doc/03-tr/缺陷训练/检测core测试集验证测试设计.md` |

# 根因分析

faster-coco-eval 的 `coco_eval.stats` 是 numpy 数组。对多元素数组做布尔判断（`array or []`）会抛上述 ValueError。评估本身已完成，失败发生在汇总写 JSON。

# 涉及文件

- `scanly/train/test_core.py`：用 `list(stats)` 取值，禁止对 numpy 数组做真值判断。

# 设计约束更新

读取 COCO `stats` / `catIds` / `iouThrs` 时按序列转换，不得使用 `value or []`。
