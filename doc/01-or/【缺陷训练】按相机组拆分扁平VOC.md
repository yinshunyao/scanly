# 背景

扁平 VOC 样本集 `cam-all-0901_voc` 在转换时去掉了 `cam1&3` / `cam2` 目录层级，复核方便但无法直接按相机组分开训练。上/下相机（1&3）与中相机（2）缺陷分布不同，合并训练会拉低分组正确率；源 YOLO 目录 `cam-all-0901` 仍按相机组存放，可用文件名（stem）回溯分组。

# 目标

1. 将扁平 VOC 按相机组拆成两套独立 VOC 目录，供分别训练。
2. 分组依据源目录 `cam1&3` 与 `cam2` 中的图像 stem，不依赖文件名末尾数字猜测。
3. 不修改原始扁平 VOC；输出到指定拆分根目录。

# 要求

## 功能性要求

1. **输入**：扁平 VOC 根（含 `JPEGImages/`、`Annotations/`）；相机对照根（含 `cam1&3/`、`cam2/`，其下缺陷子目录的 `images/`）。
2. **输出**：`came-split-0906-voc/cam1&3/` 与 `came-split-0906-voc/cam2/`，各自为完整 VOC（`JPEGImages/`、`Annotations/`，并复制 `predefined_classes.txt` 若存在）。
3. **匹配**：以图像/标注 stem 与对照目录中图像 stem 对齐；同 stem 的 jpg/xml 成对复制。
4. **冲突**：若同一 stem 同时出现在 `cam1&3` 与 `cam2`，记入报告并跳过，不得写入任一组。
5. **未匹配**：VOC 中无法对照到任一组的样本记入报告，不写入输出。

## 非功能性要求

1. 默认复制（不移动源文件）；支持 dry-run 仅统计。
2. 入口在 `scanly/train/tools/`，`__main__` 变量配置，便于 IDE 直接运行。

# 当前工作项

# 已完成工作项

## 需求

- 新增工具：按源 `cam1&3`/`cam2` 将 `cam-all-0901_voc` 拆到 `came-split-0906-voc`

## 问题

（无）
