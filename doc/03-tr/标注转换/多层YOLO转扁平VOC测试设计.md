# 关联文档

| 类型 | 文档路径 |
|:---|:---|
| 原始需求 | `scanly/doc/01-or/【标注转换】多层YOLO目录转扁平VOC.md` |
| 开发设计 | `scanly/doc/02-dr/标注转换/03.多层YOLO转扁平VOC.md` |

# 测试范围

- 多层 YOLO 包递归发现（`cam-all-0901` 形态）
- VOC 输出扁平（`JPEGImages/`、`Annotations/` 无中间层级）
- 分包 `classes.txt` 映射类别名
- 同名图像去重 / 前缀改名
- 空标签仍写 XML
- 缺图、坏标签不中断

# 测试用例

| 编号 | 前置条件 | 操作步骤 | 预期结果 | 自动化 |
|:---|:---|:---|:---|:---|
| TC-01 | `cam-all-0901` 含 `cam1&3`/`cam2` 多层 YOLO 包 | `OUTPUT_FORMAT=voc` 指向该根目录转换 | 生成扁平 `JPEGImages/` 与 `Annotations/`；无 `cam1&3/Bumps/` 子目录；xml 数量与去重后图像数一致 | 否（手工/脚本试跑） |
| TC-02 | 子包 `classes.txt` 长度不同 | 抽查 cam1 与 cam2 各一张 xml 的 `object/name` | 名称与该包 `classes.txt[id]` 一致，不是误用另一包的 id | 否 |
| TC-03 | Zigzag 与 Label 有 15 张同名同内容图 | 转换 | 只保留一份；统计 `duplicates>=15`；无覆盖损坏 | 否 |
| TC-04 | 某 txt 为空 | 转换 | 仍有对应 xml，无 `object`；negatives 增加 | 否 |
| TC-05 | 根目录无 `Categories.json`、无 YOLO 包 | 转换 | 抛出文件不存在类错误，不写半成品 | 否 |
| TC-06 | 根目录有 `Categories.json` 的客户 JSON 布局 | 转换 | 仍走原单层 JSON 逻辑，不受 YOLO 发现影响 | 否 |

# 自动化测试标识

当前交付不要求自动化用例；以 IDE 运行 `convert.py` 的 `__main__` 配置做验收。若后续要求自动化，再按 `rules-test.mdc` 在 `scanly/test/` 镜像落盘。
