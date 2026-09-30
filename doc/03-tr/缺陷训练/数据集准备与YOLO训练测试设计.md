# 关联文档

| 类型 | 文档路径 |
|:---|:---|
| 原始需求 | `scanly/doc/01-or/【缺陷训练】指定目录与类别的YOLO检测训练.md` |
| 类别角色 | `scanly/doc/01-or/【缺陷类别】核心缺陷与屏蔽非缺陷.md` |
| 开发设计 | `scanly/doc/02-dr/缺陷训练/01.数据集准备与YOLO训练.md` |

# 测试范围

- 原始目录发现（单导出包 / 多子目录如 `cam123-0820` / 扁平 VOC）
- `TRAIN_CLASSES` 过滤与跨目录 class id 重映射
- 多边形标签转 YOLO 检测框（`detect`）
- OBB 四点标签保留 / 轴对齐框展开为四顶点（`obb`）
- 过滤后空图取舍、train/val 划分、`data.yaml`
- 未知类名失败、缺图跳过、只准备不训练
- `SINGLE_CLS` 传给检测训练（不改写已预处理标签）
- `test.py` 对 `data.yaml` 的 test 划分做 val
- 连续训练自动递增 `runs/train`、`runs/train2`，不覆盖已有 run
- 检测入口默认 `epochs=200`、`patience=25` 写入 run 的 `args.yaml`

# 测试用例

| 编号 | 前置条件 | 操作步骤 | 预期结果 | 自动化 |
|:---|:---|:---|:---|:---|
| TC-01 | `cam123-0820` 各缺陷子目录齐全 | `SOURCE_DATA_ROOT` 指向该根，`TRAIN_CLASSES` 为默认 18 类，`PREPARE_ONLY=True` 跑 `train.py` | 发现 17 个子目录；输出 `images/{train,val}`、`labels/{train,val}`、`data.yaml`；标签行均为 `id cx cy w h`（5 列）；`names` 含相机 2 类（如 `BandBroken`）与相机 1/3 类（如 `GlueSeam`） | 否（脚本试跑） |
| TC-02 | 同 TC-01 | 检查 `Shorter` 子目录原 id 与 `BandBroken` 中 `Shorter` id 不同 | 输出中该类统一为 `TRAIN_CLASSES` 下标，不沿用原子目录数字 id | 否 |
| TC-03 | `TRAIN_CLASSES=["Scratch"]` | 准备数据 | 仅保留 Scratch 框；无该类的图默认不进入输出；`names` 仅 Scratch | 否 |
| TC-04 | `TRAIN_CLASSES` 含 `NotAClass` | 准备数据 | 抛错并列出原始数据中的可用类名 | 否 |
| TC-05 | 原始标签为 8 坐标多边形 | `label_format=detect` 准备后读一条 `labels/train/*.txt` | 每行 5 个数；cx/cy/w/h ∈ (0,1] | 否 |
| TC-06 | `KEEP_EMPTY_LABELS=False` 且某图过滤后无框 | 准备数据 | 该图不出现在输出 images | 否 |
| TC-07 | `PREPARE_ONLY=True` | 运行 `train.py` | 写出数据集与 `stats.json`，不调用 YOLO train | 否 |
| TC-08 | `cam123-0820` 原始为 8 坐标 OBB | `PREPARE_ONLY=True` 跑 `train_obb.py` | 输出标签每行为 9 列（`id` + 8 坐标）；`stats.label_format=obb`；四点均在 `[0,1]` | 否 |
| TC-09 | 同 TC-08 | 对比检测输出目录与 OBB 输出目录 | 二者路径不同；OBB 未把四点压成 `cx cy w h` | 否 |
| TC-10 | 合并集中 `Above` 仅出现在部分 `classes.txt` | 默认 `TRAIN_CLASSES` 准备数据 | 不把 `Above` 写入 `names`；该类框被丢弃 | 否 |
| TC-11 | 已有多类 `data.yaml`，`SKIP_PREPARE=True` | `SINGLE_CLS=True` 跑 `train.py`（可短 epochs） | 训练日志含 `single_cls=True`；run 目录 `args.yaml` 为 `single_cls: true`；不改写 `OUTPUT_DIR` 下标签与 `nc` | 否 |
| TC-12 | 同 TC-11 | `SINGLE_CLS=False`（默认）跑 `train.py` | `args.yaml` 为 `single_cls: false`；按 `data.yaml` 多类训练 | 否 |
| TC-13 | `cam-all-0901-train` 已含 `images/test` 与 `best.pt` | 按 `train.py` 对齐配置跑 `test.py`（`SPLIT=test`） | 使用 test 划分；写出 `test_metrics.json`；日志含 mAP50 | 否 |
| TC-14 | `data.yaml` 无 `test:` 或 `images/test` 为空 | 跑 `test.py` | 报错并提示 `TEST_RATIO>0` 后重新准备 | 否 |
| TC-15 | `MODEL_PATH` 指向不存在的 `.pt` | 跑 `test.py` | 报错「未找到权重」 | 否 |
| TC-16 | `runs/train` 已存在 | 再跑一次 `train.py`（可短 epochs） | 新结果写入 `runs/train2`（或下一个空号），不覆盖已有 `train` | 否 |
| TC-17 | `VOC_XML_PATH` 指向含 `JPEGImages/`、`Annotations/`、`predefined_classes.txt` 的 VOC 根 | `SKIP_PREPARE=False`、`PREPARE_ONLY=True`，`OUTPUT_DIR` 与 VOC 根不同 | 写出 YOLO `images/{train,val,test}`、`labels/*.txt`（5 列归一化）、`data.yaml`；不改写 VOC 源；`stats.source_kind=voc` | 否 |
| TC-18 | 同 TC-17，某 xml 的 `object/name` 不在 `TRAIN_CLASSES` | 准备数据 | 该框丢弃；过滤后无框且 `KEEP_EMPTY_LABELS=False` 则该图不进入输出 | 否 |
| TC-19 | `OUTPUT_DIR` 与 `VOC_XML_PATH` 为同一路径 | 准备数据 | 报错，提示不得与源目录相同 | 否 |
| TC-20 | 默认 `train.py` 超参 | 开训后读 run 目录 `args.yaml` | `epochs: 200`、`patience: 25` | 否 |

# 自动化测试标识

当前交付不要求自动化用例；以 IDE 运行 `scanly/train/train.py` / `train_obb.py`（`PREPARE_ONLY=True`）及 `test.py`（已有 test 划分与权重时）做验收。若后续要求自动化，再按 `rules-test.mdc` 在 `scanly/test/test_train/` 镜像落盘。
