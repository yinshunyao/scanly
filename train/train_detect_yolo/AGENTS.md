# train_detect_yolo

YOLO11 检测训练配置。数据源与类别只改 `../train_detect_cfg/`。训练超参改本目录 `train_config.json`。入口脚本：`../train.py`。

## 配置约定

- 公共：`train_detect_cfg/data_cfg.json`
- 本目录：`output_dir` / `skip_prepare` / `model_path` / `train` 段（imgsz、epochs、batch、patience、workers）
