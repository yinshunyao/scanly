# train_detect_yolo

YOLO11 检测训练配置。数据源、类别与 `output_dir` 只改 `../train_detect_cfg/`。训练超参改本目录 `train_config.json`。入口脚本：`../train.py`。

## 配置约定

- 公共：`train_detect_cfg/data_cfg.json`（含 `output_dir`；Ultralytics `project` 亦用该目录）
- 本目录：`skip_prepare` / `model_path` / `train` 段（imgsz、epochs、batch、patience、workers、`augment` 与官方增强键、可选 `name`）；换批次时可覆盖 `output_dir`
- `augment=true` 且未写细项时，入口按 Ultralytics detect 官方默认填增强（自动增强）
