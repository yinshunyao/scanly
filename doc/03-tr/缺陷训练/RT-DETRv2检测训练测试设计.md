# 关联文档

| 类型 | 文档路径 |
|:---|:---|
| 原始需求 | `scanly/doc/01-or/【缺陷训练】RT-DETRv2检测训练.md` |
| 开发设计 | `scanly/doc/02-dr/缺陷训练/05.RT-DETRv2检测训练.md` |
| 数据准备 | `scanly/doc/02-dr/缺陷训练/01.数据集准备与YOLO训练.md` |

# 测试范围

- 数据配置与 `train.py` 对齐（VOC / YOLO 源、`TRAIN_CLASSES`、划分、`SKIP_PREPARE`）
- YOLO 检测目录转 COCO json（类别 id、框坐标、负样本图）
- `SINGLE_CLS` 只影响 COCO、不改写 YOLO 标签
- 训练入口变量运行、run 目录递增、只准备 / 只验证
- `fit` 结束后自动 val，再按 `TEST_RATIO` 与 test 图像决定是否 test
- 不引用 scanly 以外工程的训练代码

# 测试用例

| 编号 | 前置条件 | 操作步骤 | 预期结果 | 自动化 |
|:---|:---|:---|:---|:---|
| TC-01 | 扁平 VOC 根可用，`OUTPUT_DIR` 与源不同 | `SKIP_PREPARE=False`、`PREPARE_ONLY=True` 跑 `train_rtdetrv2.py` | 写出 YOLO `data.yaml` 与 `images/{train,val,test}`；不启动 RT-DETR fit | 否 |
| TC-02 | 同 TC-01 已有 YOLO 目录 | `SKIP_PREPARE=False`、`PREPARE_ONLY=False`，短 epochs 或先只检查 COCO | `OUTPUT_DIR/coco/annotations/instances_train.json` 存在；`categories` 与 `TRAIN_CLASSES` 一致；`category_id` 从 0 计 | 否 |
| TC-03 | YOLO 某图过滤后无框且 `KEEP_EMPTY_LABELS=True` | 转 COCO | 该图在 `images` 中；`annotations` 无对应框 | 否 |
| TC-04 | 已有多类 YOLO `data.yaml` | `SINGLE_CLS=True`、`SKIP_PREPARE=True` 转 COCO（可短跑） | YOLO txt 未改；COCO `categories` 仅 1 类；全部框 `category_id=0` | 否 |
| TC-05 | `OUTPUT_DIR` 无 `data.yaml` | `SKIP_PREPARE=True` | 报错提示重新准备 | 否 |
| TC-06 | `runs/rtdetrv2` 已存在 | 再开一次新训练（非 resume） | 新目录为 `rtdetrv2_2`（或下一空号），不覆盖旧 run | 否 |
| TC-07 | 有 `best.pth` | `TEST_ONLY=True`、`RESUME` 指向该文件 | 只评估，不写新编号训练目录 | 否 |
| TC-08 | 入口源码 | 检查 import | 不出现 scanly 以外工程训练目录的 import | 否 |
| TC-09 | 同目录 `prepare_dataset` 无 `test_ratio` 形参 | `SKIP_PREPARE=False` 跑入口 | 不抛 `unexpected keyword argument 'test_ratio'`；日志警告已忽略该参数；仍写出 train/val | 否 |
| TC-10 | 短训结束，`TEST_RATIO>0` 且 `images/test` 有图 | 跑完 `train_rtdetrv2.py`（非 `TEST_ONLY`） | 日志先 val 后 test；run 目录有 `val_metrics.json` 与 `test_metrics.json`；评估后再按开关导出 ONNX | 否 |
| TC-11 | `TEST_RATIO=0` | 跑完训练 | 有 `val_metrics.json`；日志声明跳过 test；不写 `test_metrics.json` | 否 |
| TC-12 | `TEST_RATIO>0` 但 `images/test` 为空 | 跑完训练 | val 完成；日志警告跳过 test；训练不因此失败 | 否 |

# 自动化测试标识

当前交付不要求自动化用例。验收以 IDE 运行 `scanly/train/train_rtdetrv2.py`（`PREPARE_ONLY=True` 检查数据，再短训或只转 COCO）为准。若后续要求自动化，按 `rules-test.mdc` 落在 `scanly/test/test_train/`。
