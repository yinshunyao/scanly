# Scanly — 家具封边缺陷检测

English: [README.md](README.md)

训练与推理工程，用于在家具板件封边产线上**检测并识别封边相关缺陷**。样本数据、训练输出、权重与第三方二进制默认不入库，见仓库根目录 `.gitignore`。

## 目标

对封边后的家具板件图像定位缺陷区域并分类，支撑产线 NG / 贴标 / 排出判定。能力覆盖：

1. **检测** — 找出缺陷框（YOLO、检测 core / RT-DETR、检测 mark、OBB）。
2. **识别** — 按下方封边类别体系给出类名。
3. **边端推理** — HTTP（及 Gradio）服务按方案规则（面积阈值、启用开关等）输出板级结果。

现场三路线扫相机：**上** `cam1`、**中** `cam2`、**下** `cam3`。

## 已支持类别

英文名与训练 `classes.txt` / Label Studio 取值一致。角色划分见 `doc/01-or/【缺陷类别】核心缺陷与屏蔽非缺陷.md` 与 `doc/00-客户资料/业务相关/标签分类-中英文对照.md`。

### 核心缺陷（对外报出，参与 NG）

| 英文 | 中文 | 说明 |
|:---|:---|:---|
| `Tackless` | 开胶 | 封边带未粘住 / 基材外露 |
| `GlueSeam` | 胶缝 | 封边带与饰面贴合不严 |
| `GlueGap` | 崩缺 | 边角崩裂、缺料 |
| `GlueResidue` | 残胶 | 多余胶水留在带面或板面 |
| `ClampGlue` | 夹胶 | 夹具/压辊挤出的胶团（方案可默认关闭） |
| `Longer` | 过长 | 端头封边带突出未齐平 |
| `Longer2` | 过长2 | 与过长同业务、不同外观形态 |
| `Shorter` | 短带 | 封边带过短、端头露底 |
| `Below` | 封边过低 | 封边带低于饰面高度 |
| `BandBroken` | 封边带损伤 | **确定**损伤（通常可见破损面） |
| `Bumps` | 鼓包 | 封边带局部鼓起 |
| `Scrape` | 刮板 | 刮刀留下的刮痕/压痕 |

### 屏蔽非缺陷（可训练；推理默认不报出）

| 英文 | 中文 | 说明 |
|:---|:---|:---|
| `ResidualTape` | 残带 | 加工残渣/碎片（不是损伤） |
| `Hole` | 钻孔 | 工艺孔 |
| `Zigzag` | 锯齿 | 裁切锯齿边 |
| `Label` | 标签 | 板件上的贴纸/标签 |

### 其它（非核心验收；稀有或方案默认关闭）

| 英文 | 中文 | 说明 |
|:---|:---|:---|
| `BandBroken2` | 封边带损伤2 | **疑似**封边带损伤 |
| `Above` | 封边过高 | 封边带高于饰面（方案常关；默认不训练） |
| `Scratch` | 划伤 | 样本极少；表面伤以 `Scrape` 为准 |

### 当前训练焦点（v3.3.x）

`train/train_detect_cfg/data_cfg.json` 默认 `train_classes` 为三类：**`GlueSeam`（胶缝）**、**`Tackless`（开胶）**、**`GlueGap`（崩缺）**。其余类仍在类别体系与样本集中，需要时改 `train_classes` 即可启用。样本统计见 `样本数据/样本说明.md`。

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
| `样本数据/` | 本地样本（不入库） |

## 训练

改数据：`train/train_detect_cfg/data_cfg.json`  
改超参：对应入口目录下的 `train_config.json`

```bash
cd train
python train.py          # YOLO
python train_core.py     # 检测 core
python export_core.py    # 检测 core .pth → .onnx
python test_core.py      # 检测 core .pth / .onnx 评估（ONNX batch=1）
python train_mark.py     # 检测 mark
python test_mark.py      # 检测 mark test 画框（vis/ 与 test_summary.json）
python train_obb.py      # OBB
```

运行说明：`doc/99-ops/train.md`。推理运维：`doc/99-ops/run.md`。

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
/home/beyond/.conda/envs/yolo11/bin/python3 -m pyfernet run train_core.enc -e export_core.py
/home/beyond/.conda/envs/yolo11/bin/python3 -m pyfernet run train_core.enc -e test_core.py

# 仍可另打密文包（同一源目录，不同默认入口）
/home/beyond/.conda/envs/yolo11/bin/python3 -m pyfernet encrypt . -o export_core.enc -e export_core.py
/home/beyond/.conda/envs/yolo11/bin/python3 -m pyfernet run export_core.enc
```

有明文源码的机器也可直接 `python export_core.py`。

口令不要写在 `-p` 或命令行里（会进 `ps` / shell 历史）。后台跑用环境变量：

```bash
read -s PYFERNET_PASSWORD
export PYFERNET_PASSWORD

nohup /home/beyond/.conda/envs/yolo11/bin/python3 -m pyfernet run train_core.enc \
  --password-env PYFERNET_PASSWORD > scanly.log 2>&1 &

nohup /home/beyond/.conda/envs/yolo11/bin/python3 -m pyfernet run train_core.enc -e train_mark.py \
  --password-env PYFERNET_PASSWORD > scanly.log 2>&1 &

unset PYFERNET_PASSWORD
```

查看包内文件：

```bash
python3 -m pyfernet list train_core.enc
```

密文包不含 pip 依赖；客户机需自行安装 `train/requirements.txt`、`train/detect_core/requirements.txt` 与 `pyfernet-payload`。
