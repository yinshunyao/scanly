# 缺陷训练设计子目录

## 目录用途

存放封边缺陷检测训练的数据准备与训练入口设计，覆盖原始导出目录发现（含扁平 VOC）、类别过滤、格式转换、Ultralytics YOLO 训练，以及 RT-DETRv2（PyTorch）训练。

## 文档范围

**允许放入：** 原始目录约定、类别过滤与 id 重映射、**类别角色（核心缺陷 / 屏蔽非缺陷）**、多边形转检测框、扁平 VOC XML 转检测框、输出目录结构、YOLO / RT-DETRv2 训练超参入口、测试集验证、ONNX 导出与加密。

**不应放入：** 客户截图原文（放 `doc/00-客户资料/`）；测试用例表（放 `doc/03-tr/缺陷训练/`）；启动命令细节（放 `doc/99-ops/`）。

## 编辑规则

- 命名：`编号.主题.md`
- 须含 `# 关联文档`
- 变更直接改对应章节；单文档建议不超过 300 行

## 协作约定

| 相邻目录 | 关系 |
|:---|:---|
| `doc/01-or/` | 需求真源 |
| `doc/03-tr/缺陷训练/` | 同步测试设计 |
| `doc/02-dr/缺陷推理/` | 训练产出权重供推理加载 |
| `scanly/train/` | 本设计的实现落点（含配置目录 `train_detect_cfg` / `train_detect_yolo` / `train_detect_core` / `train_detect_obb`，以及 `convert_2_onnx.py`、`train_rtdetrv2.py`、`test_rtdetrv2.py`） |
