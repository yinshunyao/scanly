# 问题描述

客户机运行 `python3 test_rtdetrv2.py` 立即退出：

```
ImportError: cannot import name 'build_eval_yaml_update' from 'train_rtdetrv2'
```

现场只同步了 `test_rtdetrv2.py`，`train_rtdetrv2.py` 仍是训练入口当时的版本，没有 `build_eval_yaml_update` / `ensure_split_coco`。

# 关联文档

| 类型 | 文档路径 |
|:---|:---|
| 原始需求 | `scanly/doc/01-or/【缺陷训练】RT-DETRv2测试集验证.md` |
| 开发设计 | `scanly/doc/02-dr/缺陷训练/06.RT-DETRv2测试集验证.md` |
| 测试设计 | `scanly/doc/03-tr/缺陷训练/RT-DETRv2测试集验证测试设计.md` |

# 根因分析

评估脚本从 `train_rtdetrv2` 导入了**后来才加到训练入口**的辅助函数。现场训练脚本未同步时，import 阶段即失败，无法进入评估。

# 涉及文件

- `scanly/train/test_rtdetrv2.py`：COCO 写出与 YAML 覆盖项改在本文件内实现；只依赖训练入口已有的 `RTDETR_ROOT`、`read_class_names`、`solver_device`、`yolo_split_to_coco`。
- `scanly/train/train_rtdetrv2.py`：删除仅供测试脚本使用的 `ensure_split_coco` / `build_eval_yaml_update`，避免再次绑死双文件同步。

# 设计约束更新

`test_rtdetrv2.py` 不得依赖训练入口在评估需求之后新增的符号；只同步该测试脚本时须能 import 成功。
