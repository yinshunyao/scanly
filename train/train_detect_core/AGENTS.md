# train_detect_core

检测 core 训练配置。数据源、类别与 `output_dir` 只改 `../train_detect_cfg/`。训练超参、yaml、预训练改本目录 `train_config.json`。入口脚本：`train_core.py`（本目录）。

## 配置约定

- 公共：`train_detect_cfg/data_cfg.json`（含 `output_dir`）
- 本目录：`model_yml` / `tuning` / `run_prefix` / `skip_prepare` / `train` 段
- 训练 run 写在 `output_dir/<run_prefix>`（自动编号）
- 实现树：`../detect_core/`（YAML 在 `configs/core/`）
