# 背景

封边缺陷样本常按「相机 / 缺陷类型」多层存放，例如 `样本数据/cam-all-0901/cam1&3/Bumps/{images,labels}`。现有 `convert.py` 只识别单层客户 JSON 布局（根目录 `Categories.json` + 同层或 `labelInfo/` 下的 `*.json`），无法一次转换整棵目录树。复核需要 Pascal VOC（LabelImg），且希望转出后去掉相机与缺陷子目录，图像与 XML 都落在同一层，避免再按层级逐个打开。

# 目标

1. 支持将多层 YOLO 导出目录（每包含 `images/` 与 `labels/`）一次性转为 Pascal VOC。
2. 将 VOC 的图像与标注扁平写入 `JPEGImages/` 与 `Annotations/`，不再保留输入侧层级。
3. 按各子包自己的 `classes.txt` 把类别 id 映射为名称，避免不同子目录类别表不一致导致错类。

# 要求

## 功能性要求

1. **输入**：数据根可为单个 YOLO 包，或含多个 YOLO 包的多层目录（如 `cam-all-0901`）。YOLO 包判定：同时存在 `images/` 与 `labels/`。
2. **输出**：Pascal VOC，目录为 `JPEGImages/`、`Annotations/`、`predefined_classes.txt`；图像文件名与 XML 文件名（stem）一一对应，均在对应目录根下，不再复制 `cam1&3/Bumps/` 这类中间层级。
3. **类别**：每个子包用该包 `classes.txt`（行序即 id）把 txt 中的 class id 转为名称再写入 VOC `object/name`。`predefined_classes.txt` 为各包类别名的稳定并集。
4. **坐标**：YOLO 归一化 `cx cy w h` 按图像真实宽高换算为 VOC 像素框，并裁剪到图像范围内。
5. **扁平重名**：不同子目录出现同名图像时，若内容相同则只保留一份；若内容不同则用相对路径前缀区分文件名，不得覆盖。
6. **图像**：可选择复制到 `JPEGImages/`；XML 中 `path` 指向输出后的图像路径。
7. **负样本**：对应 txt 为空或不存在有效框时，仍写出无 `object` 的 XML。

## 非功能性要求

1. 单张失败（缺图、无法读尺寸、无法解析标签）记入 skipped，不中断整批。
2. 不修改原始样本目录。
3. 仍通过 `convert.py` 的 `__main__` 变量配置输入/输出/是否拷图，便于 IDE 直接运行。

# 当前工作项

# 已完成工作项

## 需求

- 扩展 `scanly/defects/tools/convert.py`：递归发现多层 YOLO 包，VOC 输出扁平化

## 问题

（无）
