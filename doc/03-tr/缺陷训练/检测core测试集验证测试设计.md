# 关联文档

| 类型 | 文档路径 |
|:---|:---|
| 原始需求 | `scanly/doc/01-or/【缺陷训练】检测core测试集验证.md` |
| 开发设计 | `scanly/doc/02-dr/缺陷训练/06.检测core测试集验证.md` |
| 训练设计 | `scanly/doc/02-dr/缺陷训练/05.检测core训练.md` |

# 测试范围

- 已预处理 YOLO 目录的 test/val 划分检查
- RT-DETRv2 `.pth` 与导出 `.onnx` 的 COCO mAP
- 权重类型错误（YOLO `.pt`、缺失文件）
- 指标文件写出位置不覆盖训练 run

# 测试用例

| 编号 | 前置条件 | 操作步骤 | 预期结果 | 自动化 |
|:---|:---|:---|:---|:---|
| TC-01 | `OUTPUT_DIR` 含 `images/test`，存在 `best.pth` | `SPLIT=test`、`MODEL_PATH` 指向该 `.pth` 跑 `test_core.py` | 写出 `test_metrics.json`；含 `backend=pth`、`map50`、`map50_95`；日志打印 mAP | 否 |
| TC-02 | 同划分且存在训练导出 `best.onnx` | `MODEL_PATH` 改指向 `.onnx` | `backend=onnx`；写出同样结构的指标文件 | 否 |
| TC-03 | `data.yaml` 无 `test:` 或 `images/test` 为空 | 跑入口 | 报错并提示 `TEST_RATIO>0` 后重新准备 | 否 |
| TC-04 | `MODEL_PATH` 文件不存在 | 跑入口 | 报错「未找到权重」 | 否 |
| TC-05 | `MODEL_PATH` 为 YOLO `.pt` | 跑入口 | 报错提示改用 `test.py` | 否 |
| TC-06 | val 划分非空 | `SPLIT=val`、权重为 `.pth` | 评估 val；默认仍写 `test_metrics.json`，其中 `split=val` | 否 |
| TC-07 | `SINGLE_CLS` 与训练不一致（人为设反） | 短跑或对照训练 log | 类数/指标与训练 val 对不上；验收时须与训练开关一致 | 否 |
| TC-08 | 现场 `train_core.py` 无 `build_eval_yaml_update` | 只更新 `test_core.py` 后 import / 启动 | 不报 `cannot import name 'build_eval_yaml_update'` | 否 |
| TC-09 | val 已打印 AP 行，`coco_eval.stats` 为 numpy 数组 | 写 `test_metrics.json` | 不报数组真值 ambiguous；文件含 `map50` / `map50_95` | 否 |

# 自动化测试标识

当前交付不要求自动化用例。验收以 IDE 运行 `scanly/train/test_core.py`（已有 test 划分与 `best.pth` / `best.onnx`）为准。若后续要求自动化，按 `rules-test.mdc` 落在 `scanly/test/test_train/`。
