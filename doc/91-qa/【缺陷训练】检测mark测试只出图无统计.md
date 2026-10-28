# 问题描述

检测 mark 的 test 画框图已经落盘，run 目录里没有可用的统计文件。总 TP/FP/FN、按类计数和「至少中一条」的图片数都没有。

# 关联文档

| 类型 | 文档路径 |
|:---|:---|
| 原始需求 | `scanly/doc/01-or/【缺陷训练】检测mark训练.md` |
| 开发设计 | `scanly/doc/02-dr/缺陷训练/08.检测mark训练.md` |
| 测试设计 | `scanly/doc/03-tr/缺陷训练/检测mark训练测试设计.md` |

# 根因分析

画框在 `run_test_vis` 里按 `vis_conf` / `vis_iou` 匹配后写 jpg。统计文件 `test_metrics.json` 另走运行库的评估接口；接口不存在或调用失败时只打日志并返回，不写文件。`vis/vis_summary.json` 原先只有合计，且写在图片循环之后，循环抛错时连这份也不落盘。

# 涉及文件

- `scanly/train/train_mark.py`：用画框的同一套匹配写 `test_summary.json`（run 根目录）和 `vis/vis_summary.json`。含总计数、precision、recall、`images_hit`，以及 `per_class`。循环中断仍先写出截至当前的统计。

# 设计约束更新

test 统计以匹配结果为准，不依赖运行库评估。`test_metrics.json` 仍只在评估成功时出现，不能代替 `test_summary.json`。
