# train_detect_cfg

检测训练共用数据源与类别表。YOLO（`train_detect_yolo`）/ 检测 core（`train_detect_core`）/ OBB（`train_detect_obb`）overlay 本目录，不在各入口再写一份源路径与 `train_classes`。训练超参在各入口 `train_config.json`。

## 文件

| 文件 | 用途 |
|:---|:---|
| `data_cfg.json` | 数据根、VOC、类别、划分、空标签与拷贝开关 |
| `load_cfg.py` | `merge_model_cfg`：公共 JSON + 入口 JSON |

改数据路径或类别只改本目录。入口可覆盖公共数据键。`source_data_root` / `voc_xml_path` 相对本目录解析。
