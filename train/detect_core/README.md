# detect_core

封边缺陷训练入口在上一级：`scanly/train/train_core.py`（数据配置对齐 `train.py`）。本目录是自包含的检测 core PyTorch 实现，不要从本目录外的其他工程 import。

```bash
python ../train_core.py
```

Scanly 入口 yaml 在 `configs/core/core_*_scanly.yml`。依赖见 `requirements.txt`。COCO 预训练按 `train_detect_core/train_config.json` 的 `tuning_url` 下载到 `pretrained/`。
