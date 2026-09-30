# train_detect_core

检测 core 训练配置。数据源与类别只改 `../train_detect_cfg/`。训练超参、yaml、预训练改本目录 `train_config.json`。入口脚本：`../train_core.py`。

## 配置约定

- 公共：`train_detect_cfg/data_cfg.json`
- 本目录：`model_yml` / `tuning` / `run_prefix` / `output_dir` / `train` 段
- 实现树：`../detect_core/`（YAML 在 `configs/core/`）
