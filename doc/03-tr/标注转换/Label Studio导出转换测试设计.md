# 关联文档

| 类型 | 文档路径 |
|:---|:---|
| 原始需求 | `scanly/doc/01-or/【标注转换】Label Studio导出转YOLO与LabelImg.md` |
| 开发设计 | `scanly/doc/02-dr/标注转换/01.Label Studio导出转换.md` |

# 测试范围

- Label Studio 导出目录（含递归子目录）/ 单文件 JSON 解析
- `rectanglelabels` 百分比坐标换算
- `yolo` / `voc` 输出开关与目录结构
- 仅配置输入目录 + 输出目录即可跑通
- 缺图、空标注、未知类别时的跳过与统计

# 测试用例

| 编号 | 前置条件 | 操作步骤 | 预期结果 | 自动化 |
|:---|:---|:---|:---|:---|
| TC-01 | 有含 `rectanglelabels` 的 LS JSON + 对应本地图 | `OUTPUT_FORMAT=yolo`，设 `INPUT_DIR`/`OUTPUT_DIR` 运行 | 生成 `data.yaml`、`images/train/*`、`labels/train/*.txt`；行格式为 `id cx cy w h` | 否（手工/脚本试跑） |
| TC-02 | 同 TC-01 | `OUTPUT_FORMAT=voc` 运行转换 | 生成 `JPEGImages/`、`Annotations/*.xml`、`predefined_classes.txt`；XML 含 bndbox | 否 |
| TC-03 | JSON 无 result 框 | 转换 | 仍输出空标签文件；统计 negatives 增加 | 否 |
| TC-04 | 标注含未知类别名 | 转换 | 该框跳过并告警；其余框正常写出 | 否 |
| TC-05 | 图像文件缺失 | 转换 | 该条 skipped，整批不中断 | 否 |
| TC-06 | 输入根下多子目录各含 project JSON + 图（如 `CAM1_GlueSeam600张/03`） | `RECURSIVE=True` 转换 | 各子目录任务均被收集；图按同目录/递归匹配成功 | 否 |

# 自动化测试标识

当前交付不要求自动化用例；以 IDE 运行 `convert_from_ls.py` 的 `__main__` 配置做验收。若后续要求自动化，再按 `rules-test.mdc` 在 `scanly/test/` 镜像落盘。
