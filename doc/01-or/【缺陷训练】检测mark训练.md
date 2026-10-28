# 背景

封边缺陷检测已有 YOLO（`train.py`）和检测 core（`train_core.py`），二者共用 `train_detect_cfg` 的数据源、类别与划分。需要再增加一路检测训练，用同一份公共数据对比检出，但训练目录、配置、日志和报错不能写出这路所依赖的上游实现名称。

# 目标

1. 在 `scanly/train/` 提供检测 mark 训练入口，IDE 改配置即可运行。
2. 复用公共数据配置准备 YOLO 检测目录，不另维护一份源路径或类别表。
3. 把规格与训练超参放在独立入口配置中，与数据配置解耦。
4. 保证训练代码、配置注释、运行日志和缺库报错不出现上游实现名称。
5. test 画框时同时给出推理时延。

# 要求

## 功能性要求

1. **公共数据**：入口默认读取 `scanly/train/train_detect_mark/train_config.json`，并与 `scanly/train/train_detect_cfg/data_cfg.json` 合并。源路径、`train_classes`、划分、`output_dir` 以公共配置为底；入口只在需要时覆盖同名键。数据准备调用现有 `prepare_dataset`（`label_format=detect`）。
2. **训练配置**：本入口写 `size`（`n` / `s` / `m` / `l` / `b`）、`model_path`（本地预训练权重，空则用该档自带权重）、`run_prefix`、`resume`、`device`、`single_cls`、准备开关、`test_only`、`save_vis`，以及 `train` 段（`imgsz`、`epochs`、`batch`、`grad_accum_steps`、`patience`、`workers`、`lr`、`weight_decay`、`checkpoint_interval`）。`__main__` 只保留配置路径。`model_path` 须与 `size` 同一档。
3. **数据读法**：训练直接使用准备结果中的 YOLO `data.yaml`、`images/`、`labels/`，不再为这路另写一套标注文件。`imgsz` 须能被该档的分块边长整除：`n` / `s` / `m` / `l` 为 32，`b` 为 56。默认 `size=s`、`imgsz=1024`，与检测 core / YOLO 的输入边长对齐。
4. **输出**：权重写到 `output_dir/<run_prefix>`，目录已存在则自动编号（`mark_s`、`mark_s_2`…）。`resume` 非空时写入该检查点所在目录，不新开号。`patience>0` 时按验证指标早停。
5. **评估清单**：公共配置可写 `val_classes`。未写、`null` 或空列表时，test 评估使用 `train_classes`。非空时必须是 `train_classes` 的子集，只影响 test 画框和计数；训练仍用 `train_classes`。
6. **test 画框**：训练结束后，若 `images/test` 有图且 `save_vis=true`，把画框图、`predictions.jsonl`（含匹配置信度和 IoU）和 `vis_summary.json` 写到该次 run 目录的 `vis/`（绿为对上的预测，红为多出来的预测，橙为漏掉的真值，框旁有类名和置信度；多条标签叠在一起时错开，文字都可读）。`test_mark.py` 或 `test_only=true` 只做这一步，不重新划分已有数据；权重在某次 run 目录里时，画框写到该目录的 `vis/`，否则写到 `output_dir/vis/`。`vis_conf` / `vis_iou` 只影响画图和这套计数。同一套匹配必须写出统计，不依赖运行库是否能评估：该次 run 根目录的 `test_summary.json` 与 `vis/vis_summary.json` 内容相同，含总 TP/FP/FN、precision、recall、至少中一条的图片数，以及按类的框计数和图片数。画框中途失败也要留下截至当前的这份统计。运行库能按 test 划分出指标时，另写 `test_metrics.json`。
7. **单类**：本入口不把多类框合并成一类。`single_cls=true` 且不是只准备数据时，直接失败并说明应保持 `false`，或在 `train_classes` 里只留一类。
8. **名称**：`scanly/train/` 下本入口的目录名、脚本名、配置、注释、日志和缺库/续训报错不得出现上游包名或产品名。运行库未安装时，报错只说明检测 mark 运行库未安装。
9. **test 推理时延**：test 画框时统计每次 predict 的时延（不含读图、画框和写文件），写入该次 `vis/vis_summary.json` 并在日志输出均值、中位数、P95、最小和最大。调用多于一次时第一次不计入。`save_vis=false` 时不另跑一遍推理。

## 非功能性要求

1. 不修改原始 `样本数据/` 与 VOC 源目录。
2. 不 import scanly 以外工程目录中的训练实现。
3. 该运行库写入 `scanly/train/requirements.txt`，随 `pip install -r` 安装。脚本、配置注释、日志和缺库报错仍不写上游包名或产品名。
4. 不导出 ONNX，不接入现有推理与加密导出。test 画框写在该次 run 目录的 `vis/`（权重不在 run 目录时写 `output_dir/vis/`）。
5. 不改变 YOLO、检测 core、OBB 的训练行为。

# 当前工作项

# 已完成工作项

## 需求

- 新增检测 mark 训练入口 `train_mark.py` 与 `train_detect_mark/train_config.json`，数据仍走公共 `data_cfg.json`
- 入口按 YOLO 准备结果训练；`size` 与 `imgsz` 分块约束写在配置与入口校验中
- 训练目录、配置与日志不写出上游实现名称
- `val_classes` 作为 test 评估清单，未配置时用 `train_classes`；`model_path` 指定预训练权重
- test 画框写到该次 run 目录的 `vis/`，入口为训练结束自动画，或 `test_mark.py`
- test 画框同时写入推理时延（`vis_summary.json` 与日志）
- 运行库写入 `scanly/train/requirements.txt`

## 问题

- test 画框改到该次 run 目录的 `vis/`，`predictions.jsonl` 写入 IoU；训练结束另写 `test_metrics.json`。原先写在数据集根，归档 run 时带不走 test 结果
- test 画框已出图，但统计依赖运行库评估，失败时没有文件。现用同一套匹配写出 `test_summary.json`（按类与整图计数），`vis_summary.json` 与它相同
