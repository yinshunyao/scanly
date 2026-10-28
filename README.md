# 封边缺陷检测（scanly）

训练与推理工程。样本数据、训练输出、权重与第三方二进制默认不入库，见根目录 `.gitignore`。

## 目录

| 路径 | 说明 |
|:---|:---|
| `train/` | 数据准备、YOLO / 检测 core / 检测 mark / OBB 训练 |
| `train/train_detect_cfg/` | 公共数据配置 |
| `train/train_detect_yolo/` | YOLO 训练超参 |
| `train/train_detect_core/` | 检测 core 训练超参 |
| `train/train_detect_mark/` | 检测 mark 训练超参 |
| `train/train_detect_obb/` | OBB 训练超参 |
| `train/detect_core/` | 检测 core 实现与 yaml |
| `defects/` | 缺陷推理（Python / Go） |
| `doc/` | 需求 / 设计 / 运维文档 |

## 训练

改数据：`train/train_detect_cfg/data_cfg.json`  
改超参：对应入口目录下的 `train_config.json`  

```bash
cd train
python train.py          # YOLO
python train_core.py     # 检测 core
python export_core.py    # 检测 core .pth → .onnx（单独）
python test_core.py      # 检测 core .pth / .onnx 评估（ONNX batch=1）
python train_mark.py     # 检测 mark
python test_mark.py      # 检测 mark test 画框（该次 run 的 vis/ 与 test_summary.json）
python train_obb.py      # OBB
```

运行说明：`doc/99-ops/train.md`。

## pyfernet 加密与运行

客户机只跑密文，磁盘不落训练 `.py`。先装工具，再打包整个 `scanly/train/`（入口 `train_core.py`）：

```bash
cd train
python3 -m pip install pyfernet-payload

# 加密：打包当前 train 目录，入口 train_core.py
# 加密前勿把 output/、runs/、pretrained/、大样本打进包（可先拷干净子集再 encrypt）
python3 -m pyfernet encrypt . \
  -o train_core.enc \
  -e train_core.py

# 前台运行：交互输入口令（终端不回显）
/home/beyond/.conda/envs/yolo11/bin/python3 -m pyfernet run train_core.enc
```

**入口默认写在密文包里**：不传 `-e` 时，`run` 执行加密时指定的脚本。包内其它 `.py`（如 `export_core.py` / `test_core.py`）可用 **`run --entry` 覆盖**（客户机需 `pyfernet-payload>=0.2.5`；旧包不必重打）：

```bash
# 同一 train_core.enc，换入口跑导出 / 测试
/home/beyond/.conda/envs/yolo11/bin/python3  -m pyfernet run train_core.enc -e export_core.py
/home/beyond/.conda/envs/yolo11/bin/python3  -m pyfernet run train_core.enc -e test_core.py

# 仍可另打密文包（同一源目录，不同默认入口）
/home/beyond/.conda/envs/yolo11/bin/python3  -m pyfernet encrypt . -o export_core.enc -e export_core.py
/home/beyond/.conda/envs/yolo11/bin/python3   -m pyfernet run export_core.enc
```

有明文源码的机器也可直接 `python export_core.py`。

口令不要写在 `-p` 或命令行里（会进 `ps` / shell 历史）。后台跑用环境变量：

```bash
read -s PYFERNET_PASSWORD
export PYFERNET_PASSWORD

# core
nohup /home/beyond/.conda/envs/yolo11/bin/python3 -m pyfernet run train_core.enc \
  --password-env PYFERNET_PASSWORD > scanly.log 2>&1 &

# 
nohup /home/beyond/.conda/envs/yolo11/bin/python3 -m pyfernet run train_core.enc -e train_mark.py \
  --password-env PYFERNET_PASSWORD > scanly.log 2>&1 &

unset PYFERNET_PASSWORD
```
查看包内文件：

```bash
python3 -m pyfernet list train_core.enc
```

密文包不含 pip 依赖；客户机需自行安装 `train/requirements.txt`、`train/detect_core/requirements.txt` 与 `pyfernet-payload`。
