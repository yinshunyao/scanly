# 背景

板材封边缺陷检测样本已在 Label Studio 中完成矩形框标注，并按系统导出得到 JSON。现有 `defects/tools/convert.py` 仅支持客户专用标注格式，无法直接消费 Label Studio 导出结果。训练侧需要 YOLO11 检测格式，人工复核侧需要 LabelImg（Pascal VOC）可读格式，二者需可切换输出。

# 目标

1. 支持将 Label Studio 导出的矩形框标注批量转换为 YOLO11 训练目录结构。
2. 支持将同一批导出转换为 LabelImg 可打开的 Pascal VOC 格式。
3. 通过配置开关在两种目标格式间切换，无需改转换主流程。
4. 入口只需配置输入目录与输出目录即可批量转换（默认可递归子目录）。

# 要求

## 功能性要求

1. **输入**：Label Studio JSON 导出，至少支持：
   - 单文件：任务数组（每项含 `data` / `annotations[].result`）
   - 目录：目录下 JSON（每文件可为任务数组或单任务）；**默认递归子目录**
2. **标注类型**：解析 `rectanglelabels`；坐标为相对百分比（`x/y/width/height`，相对原图宽高的 0–100）。
3. **输出开关**：配置项可选 `yolo` 或 `voc`（LabelImg / Pascal VOC）。
4. **YOLO 输出**：目录含 `data.yaml`、`images/train/`、`labels/train/`；标签行为 `class_id cx cy w h`（归一化）。
5. **LabelImg 输出**：Pascal VOC（`JPEGImages/`、`Annotations/`、`predefined_classes.txt`），可用 LabelImg 的 PascalVOC 模式打开复核。
6. **类别**：优先使用 `Categories.json` 固定 `id → name`；若未提供则从导出中收集标签名并稳定编号。
7. **目录配置**：入口只需 **输入目录** + **输出目录**；输入目录同时作为 JSON 与图像根（图可与 JSON 同层或在子目录中）。
8. **负样本**：无矩形结果时仍生成空标签（YOLO 空 `.txt` / VOC 无 object），便于训练包含负例。

## 非功能性要求

1. 不依赖额外重型图像库读取尺寸（与现有 `convert.py` 一致，可解析常见 bmp/jpg/png）。
2. 跳过无法匹配图像或无法解析的条目并统计，不得因单条失败中断整批。
3. 本工具为离线转换脚本，不接入在线服务。

# 当前工作项

# 已完成工作项

## 需求

- 实现 `scanly/train/tools/convert_from_ls.py`：LS 导出 → YOLO11 / LabelImg（VOC）
- 入口配置简化为输入目录 + 输出目录；默认递归子目录扫描 JSON/图像

## 问题

（无）
