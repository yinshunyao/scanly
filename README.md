# 封边缺陷检测（scanly）

训练与推理工程。样本数据、训练输出、权重与第三方二进制默认不入库，见根目录 `.gitignore`。

## 目录

| 路径 | 说明 |
|:---|:---|
| `train/` | 数据准备、YOLO / RT-DETRv2 / OBB 训练 |
| `train/train_detect_cfg/` | 公共数据配置 |
| `train/train_detect_yolo/` | YOLO 训练超参 |
| `train/train_detect_core/` | RT-DETRv2 训练超参 |
| `train/train_detect_obb/` | OBB 训练超参 |
| `defects/` | 缺陷推理（Python / Go） |
| `doc/` | 需求 / 设计 / 运维文档 |

## 训练

改数据：`train/train_detect_cfg/data_cfg.json`  
改超参：对应入口目录下的 `train_config.json`  

```bash
python train/train.py
python train/train_rtdetrv2.py
python train/train_obb.py
```

运行说明：`doc/99-ops/train.md`。
