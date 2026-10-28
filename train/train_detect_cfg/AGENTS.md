# train_detect_cfg

检测训练共用数据源、类别表与输出目录。YOLO（`train_detect_yolo`）/ 检测 core（`train_detect_core`）/ OBB（`train_detect_obb`）/ 检测 mark（`train_detect_mark`）overlay 本目录，不在各入口再写一份源路径与 `train_classes`。训练超参在各入口 `train_config.json`。

## 文件

| 文件 | 用途 |
|:---|:---|
| `data_cfg.json` | 数据根、VOC、`output_dir`、类别、划分、空标签与拷贝开关 |
| `load_cfg.py` | `merge_model_cfg`：公共 JSON + 入口 JSON |
| `prepare_dataset.py` | YOLO 检测/OBB 数据准备（VOC 或 LS 导出 → `output_dir`） |

改数据路径、输出目录或类别只改本目录。入口可覆盖公共数据键。`val_classes` 是评估关注类别：检测 core 用它选 `best.pth`，检测 mark 的 test 用它画框和计数；空列表时 core 按全部评估类别，mark 按 `train_classes`。`source_data_root` / `voc_xml_path` / `output_dir` 相对本目录解析。`voc_xml_path` 可为字符串或字符串列表。`voc_split.split_each_dir` 默认 `true`：多目录时每个目录单独划分再合并；`false` 则全部打乱后一次划分。`output_dir` 同时作为预处理 YOLO 根与训练权重输出根。
