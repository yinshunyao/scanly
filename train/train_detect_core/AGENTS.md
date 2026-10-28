# train_detect_core

检测 core 训练配置。数据源、类别与 `output_dir` 只改 `../train_detect_cfg/`。训练超参、yaml、预训练改本目录 `train_config.json`。入口脚本：`../train_core.py`（训练）、`../export_core.py`（单独转 ONNX）、`../test_core.py`（`.pth`/`.onnx` 评估；ONNX 强制 batch=1）。

## 配置约定

- 公共：`train_detect_cfg/data_cfg.json`（含 `output_dir`）
- 本目录：`backbone`（`r18` / `r34` / `r50`，自动关联 yaml）/ `tuning` / `run_prefix` / `skip_prepare` / `single_cls` / `save_vis` / `train` 段
- v3.3 主数据：`train_classes`=胶缝/开胶/崩缺（见公共 `data_cfg`）
- v3.3.5：崩缺源换 `B01B02B03B04B05B06-1028-voc`，r34 三类超参对齐 v3.3.1；staging `/root/autodl-tmp/output/v3.3.5`
- `single_cls=true`：检测不区分分类，框并成一类 `defect`（`run_prefix` 建议带 `s1`，归档 v3.3.2）；分类由后续独立模型完成。此时本目录须覆盖 `val_classes: []`
- Nano 对照（归档 v3.3.3）走 `train_detect_mark`，`size=n`
- 训练 run 写在 `output_dir/<run_prefix>`（自动编号）。test 画框在该目录的 `vis/`
- 实现树：`../detect_core/`（YAML 在 `configs/core/`）
- ONNX：训练结束可由 `export_onnx` 自动导出；补导出用 `export_core.py`。评估与部署有效 batch=1。