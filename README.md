# Scanly — Furniture Edge-Banding Defect Detection

Chinese: [README_cn.md](README_cn.md)

Training and inference for **detecting and recognizing furniture edge-banding defects** from line-scan / panel images. Samples, training outputs, weights, and third-party binaries are gitignored by default (see repo root `.gitignore`).

## Goal

On furniture panels after edge banding, locate defect regions and classify them so production can decide NG / labeling / discharge. The system covers:

1. **Detection** — find defect boxes (YOLO, detection core / RT-DETR, detection mark, OBB).
2. **Recognition** — assign class names from the edge-banding taxonomy below.
3. **Edge inference** — HTTP (and Gradio) services that apply scheme rules (area thresholds, enable flags) and return board-level outcomes.

Three line-scan cameras are used in the field: **top** `cam1`, **side** `cam2`, **bottom** `cam3`.

## Supported classes

English names match training `classes.txt` / Label Studio values. Roles follow `doc/01-or/【缺陷类别】核心缺陷与屏蔽非缺陷.md` and `doc/00-客户资料/业务相关/标签分类-中英文对照.md`.

### Core defects (report as defects; participate in NG)

| English | Chinese | Notes |
|:---|:---|:---|
| `Tackless` | 开胶 | Unbonded edge band / substrate exposed |
| `GlueSeam` | 胶缝 | Incomplete seam between band and face |
| `GlueGap` | 崩缺 | Chip-out / missing material at edge |
| `GlueResidue` | 残胶 | Excess glue on band or face |
| `ClampGlue` | 夹胶 | Glue blob from clamp/roller (may be disabled in scheme) |
| `Longer` | 过长 | Band overhang at panel end |
| `Longer2` | 过长2 | Same business as Longer, different visual form |
| `Shorter` | 短带 | Band too short; substrate exposed at end |
| `Below` | 封边过低 | Band sunk below face height |
| `BandBroken` | 封边带损伤 | Confirmed band damage (broken face usually visible) |
| `Bumps` | 鼓包 | Local bulge on the band |
| `Scrape` | 刮板 | Scraper marks / surface dents |

### Shielded non-defects (trainable; inference hides by default)

| English | Chinese | Notes |
|:---|:---|:---|
| `ResidualTape` | 残带 | Debris / scrap tape (not damage) |
| `Hole` | 钻孔 | Process drill hole |
| `Zigzag` | 锯齿 | Cut zigzag edge |
| `Label` | 标签 | Sticker / label on the panel |

### Other (not core acceptance; rare or scheme-off)

| English | Chinese | Notes |
|:---|:---|:---|
| `BandBroken2` | 封边带损伤2 | Suspected band damage |
| `Above` | 封边过高 | Band higher than face (scheme usually off; not in default train set) |
| `Scratch` | 划伤 | Rare; surface damage prefers `Scrape` |

### Current training focus (v3.3.x)

Default `train_classes` in `train/train_detect_cfg/data_cfg.json` is three classes: **`GlueSeam`**, **`Tackless`**, **`GlueGap`**. Other classes remain in the catalog and datasets; enable them via `train_classes` when needed. Sample counts: `样本数据/样本说明.md`.

## Layout

| Path | Description |
|:---|:---|
| `train/` | Data prep; YOLO / detect core / detect mark / OBB training |
| `train/train_detect_cfg/` | Shared data config |
| `train/train_detect_yolo/` | YOLO hyperparams |
| `train/train_detect_core/` | Detect-core hyperparams |
| `train/train_detect_mark/` | Detect-mark hyperparams |
| `train/train_detect_obb/` | OBB hyperparams |
| `train/detect_core/` | Detect-core implementation and yaml |
| `defects/` | Defect inference (Python / Go) |
| `doc/` | Requirements / design / ops docs |
| `样本数据/` | Local samples (not in git) |

## Training

Edit data: `train/train_detect_cfg/data_cfg.json`  
Edit hyperparams: `train_config.json` under the matching entry directory.

```bash
cd train
python train.py          # YOLO
python train_core.py     # detect core
python export_core.py    # detect core .pth → .onnx
python test_core.py      # detect core .pth / .onnx eval (ONNX batch=1)
python train_mark.py     # detect mark
python test_mark.py      # detect mark test boxes (vis/ + test_summary.json)
python train_obb.py      # OBB
```

Ops notes: `doc/99-ops/train.md`. Inference ops: `doc/99-ops/run.md`.

## pyfernet encrypt & run

Customer machines run ciphertext only (no plaintext training `.py` on disk). Install the tool, then encrypt the whole `scanly/train/` tree (entry `train_core.py`):

```bash
cd train
python3 -m pip install pyfernet-payload

# Encrypt current train dir; entry train_core.py
# Do not pack output/, runs/, pretrained/, or large samples (copy a clean subset first)
python3 -m pyfernet encrypt . \
  -o train_core.enc \
  -e train_core.py

# Foreground: password prompted (no echo)
/home/beyond/.conda/envs/yolo11/bin/python3 -m pyfernet run train_core.enc
```

**Default entry is stored in the package.** Without `-e`, `run` uses the script chosen at encrypt time. Other `.py` files in the package (e.g. `export_core.py` / `test_core.py`) can override with **`run --entry`** (`pyfernet-payload>=0.2.5`; old packages need not be rebuilt):

```bash
/home/beyond/.conda/envs/yolo11/bin/python3 -m pyfernet run train_core.enc -e export_core.py
/home/beyond/.conda/envs/yolo11/bin/python3 -m pyfernet run train_core.enc -e test_core.py

# Or encrypt separate packages with different default entries
/home/beyond/.conda/envs/yolo11/bin/python3 -m pyfernet encrypt . -o export_core.enc -e export_core.py
/home/beyond/.conda/envs/yolo11/bin/python3 -m pyfernet run export_core.enc
```

Machines with source can still run `python export_core.py` directly.

Do not put the password in `-p` or the shell history. For background runs use an env var:

```bash
read -s PYFERNET_PASSWORD
export PYFERNET_PASSWORD

nohup /home/beyond/.conda/envs/yolo11/bin/python3 -m pyfernet run train_core.enc \
  --password-env PYFERNET_PASSWORD > scanly.log 2>&1 &

nohup /home/beyond/.conda/envs/yolo11/bin/python3 -m pyfernet run train_core.enc -e train_mark.py \
  --password-env PYFERNET_PASSWORD > scanly.log 2>&1 &

unset PYFERNET_PASSWORD
```

List package contents:

```bash
python3 -m pyfernet list train_core.enc
```

The encrypted package does not include pip deps. Install `train/requirements.txt`, `train/detect_core/requirements.txt`, and `pyfernet-payload` on the customer machine.
