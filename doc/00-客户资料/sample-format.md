# 缺陷检测标注样本格式说明

## 概述

本数据集为**工业产线缺陷检测**场景下的目标检测标注，由专用标注工具导出。每张图像对应一个 JSON 标注文件，标注类型均为**水平矩形框（bbox）**，适用于 YOLO 系列检测模型训练或 LabelMe 人工复核。

完整数据集目录结构参考 `/Volumes/shunyao-h1/work/defects/samples`；仓库内 `defects/samples/` 为精简样例（1 张图 + 1 个标注 + 类别表）。

## 目录结构

### 标准数据集布局

```
samples/
├── Categories.json          # 类别定义（id、名称、显示颜色）
├── inputImages/             # 原始图像
│   ├── xxx.jpg
│   └── yyy.bmp
└── labelInfo/               # 标注 JSON，文件名 = 图像名 + .json
    ├── xxx.jpg.json
    └── yyy.bmp.json
```

### 样例目录布局（扁平）

```
samples/
├── Categories.json
├── image.bmp
└── image.bmp.json
```

转换工具 `tools/convert.py` 会自动识别以上两种布局。

## 文件说明

### Categories.json

全局类别表，字段 `categories` 为数组，每项包含：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | int | 类别 ID，从 0 开始，与标注中 `category_id` 对应 |
| `name` | string | 类别英文名 |
| `color_r/g/b/a` | int | 标注工具中的显示颜色（转换时可忽略） |

当前 13 个类别：

| id | name | 含义（据名称推断） |
|----|------|-------------------|
| 0 | GlueGap | 胶缝间隙 |
| 1 | GlueSeam | 胶缝 |
| 2 | Longer | 过长 |
| 3 | GlueResidue | 残胶 |
| 4 | Tackless | 无粘性 |
| 5 | ResidualTape | 残胶带 |
| 6 | Bumps | 凸起 |
| 7 | ClampGlue | 夹具胶 |
| 8 | Label | 标签 |
| 9 | Longer2 | 过长（二类） |
| 10 | Hole | 孔洞 |
| 11 | Scrape | 刮痕 |
| 12 | Zigzag | 锯齿/曲折 |

### 单图标注 JSON（labelInfo）

每个文件结构如下：

```json
{
  "images": [
    {
      "file_name": "xxx.jpg",
      "width": 1024,
      "height": 1024,
      "NoTarget": false,
      "mark": true,
      "imagestatus": 2
    }
  ],
  "annotations": [
    {
      "id": 0,
      "category_id": 8,
      "type": 3,
      "points": [x1, y1, x2, y2],
      "angle": 0,
      "area": 76524.65,
      "roiId": 0,
      "instanceID": "",
      "promptInfo": {}
    }
  ]
}
```

#### images 字段

| 字段 | 说明 |
|------|------|
| `file_name` | 对应图像文件名（与 `inputImages/` 中文件一致） |
| `width` / `height` | 图像尺寸（**可能与实际文件不一致**，转换时以图像文件真实尺寸为准） |
| `NoTarget` | `true` 表示无目标（负样本），此时 `annotations` 通常为空 |
| `mark` | 是否已标注 |
| `imagestatus` | 标注流程状态码（转换时可忽略） |

#### annotations 字段

| 字段 | 说明 |
|------|------|
| `category_id` | 对应 `Categories.json` 中的 `id` |
| `type` | 形状类型；当前数据集中**全部为 `3`**，表示轴对齐矩形 |
| `points` | 4 个浮点数：`[x1, y1, x2, y2]`，左上角与右下角坐标（像素） |
| `angle` | 旋转角；当前均为 `0`（无旋转） |
| `area` | 框面积（可忽略） |

> **注意**：`type=3` 且 `points` 长度为 4 时，按水平矩形框处理，与 YOLO 检测格式直接对应。

## 图像特征

基于完整数据集（1000 张）统计：

| 项目 | 数值 |
|------|------|
| 图像数量 | 1000 |
| 标注文件数量 | 1000（与图像一一对应） |
| 图像格式 | `.jpg` 922 张，`.bmp` 78 张 |
| 图像尺寸 | 均为 **1024×1024** |
| 负样本（NoTarget） | 96 张（无标注框） |
| 有框样本 | 904 张 |
| 标注框总数 | 1577 个 |
| 形状类型 | 全部为矩形（type=3） |

文件名通常包含产线信息，例如：

```
00282 02663_2024-09-20 21_58_21.003_HC_pFCu67P6Kc_82up.bmp
61104 2025-09-15 15_43_32.614_1_5575.WD000597084A1061_4_1_9_5575up.bmp
```

后缀 `up` / `bottom` 表示工位上下视角。

## 转换输出格式

使用 `defects/tools/convert.py` 可将本格式转为：

### 1. YOLO11 检测训练格式

```
output/
├── data.yaml
├── images/
│   └── train/          # 图像副本
└── labels/
    └── train/          # 同名 .txt，每行：class_id cx cy w h（归一化）
```

`data.yaml` 自动写入 `nc` 与 `names`，可直接用于 Ultralytics YOLO 训练。

### 2. LabelMe 格式

```
output/
├── xxx.jpg
└── xxx.json            # LabelMe 标准 JSON，shape_type=rectangle
```

可用 [LabelMe](https://github.com/wkentaro/labelme) 打开查看或二次编辑。

### 3. Pascal VOC 格式（LabelImg）

```
output/
├── JPEGImages/             # 图像
├── Annotations/            # 同名 .xml
└── predefined_classes.txt  # LabelImg 预定义类别
```

可用 [LabelImg](https://github.com/HumanSignal/labelImg) 打开复核：

1. 菜单选择 **PascalVOC** 格式
2. **Open Dir** 打开 `JPEGImages/`
3. **Change Save Dir** 指向 `Annotations/`（或整个 `voc/` 根目录）

### 使用示例

在 `convert.py` 底部修改变量后于 IDE 中运行，或在其他脚本中调用：

```python
from convert import convert_dataset

convert_dataset(
    input_dir="/Volumes/shunyao-h1/work/defects/samples",
    output_dir="/path/to/output/yolo",
    output_format="yolo",
    copy_images=True,
)
```

支持参数：

- `input_dir`：含 `Categories.json` 的数据根目录
- `output_dir`：输出目录
- `output_format`：`yolo`、`labelme` 或 `voc`
- `copy_images`：是否复制图像（`False` 时 YOLO 输出仅写标签与 `data.yaml`）

## 已知问题

1. **尺寸元数据偏差**：部分 JSON 中 `width`/`height` 为 2048，但实际图像为 1024×1024；转换脚本读取图像文件真实尺寸。
2. **仅矩形框**：当前数据无多边形、分割掩码；若未来出现其他 `type` 值，需扩展解析逻辑。
