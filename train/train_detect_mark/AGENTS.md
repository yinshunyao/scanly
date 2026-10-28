# train_detect_mark

检测 mark 训练配置。数据源、类别与 `output_dir` 只改 `../train_detect_cfg/`。规格与超参改本目录 `train_config.json`。入口脚本：`../train_mark.py`。

## 配置约定

- 公共：`train_detect_cfg/data_cfg.json`（含 `output_dir`）
- 本目录：`size`（`n` / `s` / `m` / `l` / `b`）/ `run_prefix` / `resume` / `skip_prepare` / `train` 段
- `train.imgsz`：`n` / `s` / `m` / `l` 须为 32 的倍数，`b` 须为 56 的倍数；本入口默认 s / 1024，与检测 core 对齐
- 训练 run 写在 `output_dir/<run_prefix>`（自动编号）
- `model_path`：本地预训练权重，须与 `size` 同一档；空则用该档自带权重
- test 画框读公共 `val_classes`（空则用 `train_classes`），写到该次 run 目录的 `vis/`；匹配统计写 `test_summary.json`（与 `vis/vis_summary.json` 相同）。运行库能评估时另写 `test_metrics.json`
- 只评估不训练：`../test_mark.py`
- 直接使用准备结果里的 YOLO `data.yaml`，不另写标注
- 本目录与入口不写上游实现名称
