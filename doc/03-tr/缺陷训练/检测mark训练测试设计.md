# 关联文档

| 类型 | 文档路径 |
|:---|:---|
| 原始需求 | `scanly/doc/01-or/【缺陷训练】检测mark训练.md` |
| 开发设计 | `scanly/doc/02-dr/缺陷训练/08.检测mark训练.md` |
| 关联设计 | `scanly/doc/02-dr/缺陷训练/07.数据配置与训练配置解耦.md` |

# 测试范围

- 入口 `train_mark.py` 读取 `train_detect_mark/train_config.json` 并 overlay 公共 `data_cfg.json`
- `prepare_only` 只准备 YOLO 目录，不要求运行库已安装
- `size` 与 `imgsz` 的分块约束在开训前失败
- `single_cls=true` 在训练路径失败，且不改写已有标签
- run 目录落在 `output_dir/<run_prefix>`，已存在则编号
- `resume` 指向文件或含检查点的目录时不新开号
- 入口目录、配置、注释、缺库报错和训练日志正文不出现上游实现名称
- 公共 `val_classes` 对 test 画框的过滤，以及缺省时回落到 `train_classes`
- `model_path` 作为本地预训练权重
- test 画框写到该次 run 目录的 `vis/`；`predictions.jsonl` 含 IoU；run 根目录有 `test_summary.json`（与 `vis/vis_summary.json` 相同，含按类与整图计数）；有评估接口时同目录还有 `test_metrics.json`
- test 画框同时写出推理时延

# 测试用例

| 编号 | 前置条件 | 操作步骤 | 预期结果 | 自动化 |
|:---|:---|:---|:---|:---|
| TC-01 | 公共 `data_cfg` 有效，入口 `prepare_only=true` | 运行 `train_mark.py` | 读 mark 配置 overlay 公共数据；写出 `output_dir` 的 `data.yaml`；不训练 | 否 |
| TC-02 | 只改公共 `train_classes` | `prepare_only` 跑 `train.py` 与 `train_mark.py` | 两侧 `classes.txt` / `names` 与公共列表一致 | 否 |
| TC-03 | 入口 `size` 写成 `x` | 运行入口 | 开训前失败，提示只能是 `n` / `s` / `m` / `l` / `b` | 否 |
| TC-04 | `size=s` 且 `train.imgsz=1000` | 运行入口 | 开训前失败，说明 s 的边长须被 32 整除 | 否 |
| TC-05 | `size=b` 且 `train.imgsz=1024` | 运行入口 | 开训前失败，说明 b 的边长须被 56 整除 | 否 |
| TC-06 | `single_cls=true`，`prepare_only=false` | 运行入口 | 失败并说明不合并多类；已有标签文件不被改写 | 否 |
| TC-07 | `single_cls=true`，`prepare_only=true` | 运行入口 | 仍可完成数据准备 | 否 |
| TC-08 | `skip_prepare=true` 且 `output_dir` 无 `data.yaml` | 运行入口 | 失败，提示先准备数据 | 否 |
| TC-09 | `skip_prepare=true` 且 `prepare_only=true` | 运行入口 | 配置错误，不训练 | 否 |
| TC-10 | `output_dir/mark_s` 已存在，`resume` 为空 | 开训 | 新目录为 `output_dir/mark_s_2`（再占则继续编号） | 否 |
| TC-11 | `resume` 指向某次 run 里的 `last.ckpt` | 开训 | 写入该检查点所在目录，不新开号 | 否 |
| TC-12 | `resume` 指向空目录 | 开训 | 失败，提示目录里没有可续训检查点 | 否 |
| TC-13 | 当前环境未安装运行库，`prepare_only=false` | 运行入口 | 报错只说明检测 mark 运行库未安装，正文不含上游包名 | 否 |
| TC-14 | 检索 `train_mark.py`、`train_detect_mark/` | 全文检索上游包名、带连字符产品名 | 无连续命中 | 否 |
| TC-15 | 训练日志含上游产品名 | 观察入口包过的标准输出 | 对应片段被替换为 `mark` | 否 |
| TC-16 | 公共 `val_classes` 为 `GlueSeam`、`Tackless`，训练类还含 `ResidualTape` | 跑 test 画框 | 图和 `predictions.jsonl` 里没有残带；训练标签文件仍保留残带 | 否 |
| TC-17 | 删掉或置空 `val_classes` | 跑 test 画框 | 评估类与 `train_classes` 一致 | 否 |
| TC-18 | `val_classes` 含不在 `train_classes` 中的名字 | 跑入口 | 开训前失败，提示必须是子集 | 否 |
| TC-19 | `model_path` 指向存在的本地 `.pth` | 开训 | 用该文件初始化，不下载该档自带权重 | 否 |
| TC-20 | `model_path` 指向不存在的路径 | 开训 | 失败，提示预训练权重不存在 | 否 |
| TC-21 | test 有图，`save_vis=true` | 训练结束或跑 `test_mark.py` | 该次 run 的 `vis/` 有画框图；绿/红/橙与类名、置信度可见；`predictions.jsonl` 的预测含 `iou`；run 根目录有 `test_summary.json`，与 `vis/vis_summary.json` 内容相同。两份都有总 `tp`/`fp`/`fn`、`precision`、`recall`、`images_hit`，以及 `per_class` 里每个评估类的框计数和 `images_hit`；`latency.scope=predict`，并有 `mean_ms` / `p50_ms` / `p95_ms`；日志有时延和按类一行。运行库不能评估时仍有 `test_summary.json`。运行库能评估 test 时，run 目录还有 `test_metrics.json` | 否 |
| TC-22 | `test_only` 且 `output_dir` 已有 `data.yaml` | 跑 `test_mark.py` | 不重新划分数据 | 否 |
| TC-23 | 同一位置有 FN 与 FP 两条标签 | 画评估图 | 两条文字背景不重叠，类名和置信度都能读 | 否 |
| TC-24 | test 多于一张图，`save_vis=true` | 看 `vis_summary.json` 的 `latency` | `warmup_excluded=1`；`images` 比画出的图少 1 | 否 |
| TC-25 | `save_vis=false` | 训练结束 | 不写 `vis/`，也不另跑推理 | 否 |

# 自动化测试标识

当前交付不要求自动化。验收方式：IDE 改 JSON 后跑入口；名称约束用全文检索核对 TC-14。
