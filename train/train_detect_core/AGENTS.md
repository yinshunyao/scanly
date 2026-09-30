# train_detect_core

RT-DETRv2 检测训练配置。数据源与类别只改 `../train_detect_cfg/`。训练超参、yaml、预训练改本目录 `train_config.json`。入口脚本：`../train_rtdetrv2.py`。

## 配置约定

- 公共：`train_detect_cfg/data_cfg.json`
- 本目录：`model_yml` / `tuning` / `run_prefix` / `output_dir` / `train` 段
