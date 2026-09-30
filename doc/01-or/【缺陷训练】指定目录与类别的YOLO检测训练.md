# 背景

封边缺陷样本按采集批次落在 `scanly/样本数据/` 下。当前默认批次 `cam123-0820` 为相机 1/2/3 合并后的 Label Studio 导出：每个缺陷一个子目录，内含 `images/`、`labels/`、`classes.txt`。各子目录的类别表并不一致（同一英文类名对应的数字 id 可能不同；相机 2 子包含 `Above`/`BandBroken` 等，相机 1/3 子包类表更短），标签行多为归一化四边形（8 个坐标，OBB / 旋转框风格）。复核后的批次常以 LabelImg 扁平 Pascal VOC 交付（`JPEGImages/`、`Annotations/*.xml`、`predefined_classes.txt`，如 `cam2-0911-1024`）。直接拿原始 YOLO 目录训练会类别错位；VOC 则不能被 Ultralytics 直接读取。训练前必须能指定原始数据根目录和要引入的类别，按类名过滤并按任务写出对应格式。

# 目标

1. 指定原始数据集目录，自动识别 Label Studio YOLO 导出布局，或扁平 Pascal VOC（`JPEGImages` + `Annotations`）。
2. 指定要引入训练的类别列表，仅保留这些类的框，并按该列表重排 class id。
3. 支持两类训练数据准备：检测（多边形→轴对齐框）与 OBB（保留/规范四点），合并子目录、划分 train/val/test、写出 `data.yaml`。
4. 在 `scanly/train/` 分别提供 YOLO11 检测（`train.py`）与 OBB（`train_obb.py`）训练入口。
5. 支持将已纳入训练的多类框按单类训练（只关心有无目标）。
6. 对已划分的测试集独立验证训练权重，配置与检测训练入口对齐。

# 要求

## 功能性要求

1. **原始目录**：配置项指定数据集根路径（默认适配 `scanly/样本数据/cam123-0820`）。根目录本身是导出包，或根下多个导出子目录，均须支持。检测入口另提供 `VOC_XML_PATH`：指向扁平 VOC 根（含 `JPEGImages/`、`Annotations/`，以及可选 `predefined_classes.txt`）；非空时优先于 YOLO 源目录。
2. **训练类别**：配置项为英文类名列表（与 `classes.txt` / `notes.json` / VOC `object/name` 一致）。只训练列表中的类；其它框丢弃。列表顺序即为训练 class id（从 0 起）。默认列表为相机 1/2/3 合并后的有效类（不含 `Above`：方案关闭且样本极少）。
3. **跨子目录对齐**：各子目录用各自的 `classes.txt` 把数字 id 还原为类名，再映射到配置的训练类别 id，禁止跨目录直接复用数字 id。
4. **检测格式**：标签若为多边形（坐标点数 ≥ 6），取外接轴对齐框转为 YOLO 检测行；若已是 4 个数的检测框则保留。VOC `bndbox` 像素框转为归一化 `id cx cy w h`。写出标准 YOLO11 检测目录。转换结果不得写回 VOC 源目录。
5. **OBB 格式**：标签若已是四点（8 坐标）则保留为 YOLO OBB 行 `id x1 y1 x2 y2 x3 y3 x4 y4`；若为轴对齐检测框则展开为矩形四顶点。写出标准 YOLO11 OBB 目录。本需求不要求同步改造推理服务。
6. **样本取舍**：过滤后无剩余框的图像默认不进入训练集；可配置是否保留空标签负样本。
7. **划分**：按子目录分层划分 train/val/test，比例可配置。样本过少时允许 val 回退为 train。
8. **训练**：使用 Ultralytics YOLO；检测入口 `task=detect`，OBB 入口 `task=obb`。入口参数放在 `__main__` 变量区（IDE 可直接运行），不强制 argparse。可只做数据准备、不启动训练。检测训练不要求配置 run 子目录名；`TRAIN_PROJECT` 下已有输出时由 Ultralytics 自动递增（`train` / `train2` / …），不覆盖已有 run。检测入口默认 `EPOCHS=200`（上限）、`PATIENCE=25`（val fitness 连续 25 epoch 不刷新 `best.pt` 则早停）。
9. **单分类训练**：检测入口提供 `SINGLE_CLS`（默认关）。开启后，将已纳入训练的全部框视为同一类（Ultralytics `single_cls`），用于只关心「有无目标」而不区分类型。关闭时保持多类。不改写已预处理标签文件；`SKIP_PREPARE=True` 时同样生效。类别过滤仍由 `TRAIN_CLASSES` 决定。
10. **测试集验证**：检测入口提供 `test.py`，配置项与 `train.py` 对齐（数据目录、imgsz、batch、device、single_cls）。默认对 `data.yaml` 的 `test` 划分调用 Ultralytics `val`，输出 mAP 等指标；无测试集或权重缺失时报错，提示重新准备或核对路径。不强制 argparse。
11. **ONNX 导出**：训练完成后可用同目录 `convert_2_onnx.py` 将 `.pt` 导出为 ONNX 并加密，供 Go 边端加载。变量入口，不强制 argparse。

## 非功能性要求

1. 单条坏标签或缺图不得中断整批；跳过并统计。
2. 不修改原始数据集目录；转换结果写到独立输出目录（检测与 OBB 输出目录宜分开）。
3. 训练依赖：Ultralytics YOLO11（含 OBB 预训练权重）。

# 当前工作项

# 已完成工作项

## 需求

- 创建 `scanly/train` 训练代码：指定原始目录与训练类别、过滤、必要时转换，并启动 YOLO11 检测训练
- 新增 `train_obb.py`：在相同数据配置能力下支持 OBB 数据集准备与 YOLO11-OBB 训练（暂不改推理）
- 默认训练数据切到合并批次 `cam123-0820`，类别列表覆盖相机 1/2/3 有效类（不含 `Above`）
- 新增 `convert_2_onnx.py`：将训练 `.pt` 导出为加密 ONNX
- `train.py` 增加 `SINGLE_CLS`：开启后多类框按单类训练，默认关闭
- 新增 `test.py`：对齐 `train.py` 配置，对已划分测试集做 Ultralytics val
- `train.py` 去掉 `RUN_NAME`：不覆盖已有 run，由 Ultralytics 自动递增目录
- `train.py` 支持 `VOC_XML_PATH`：扁平 VOC（JPEGImages + Annotations）转为 YOLO 后再训练
- 检测入口默认改为 `EPOCHS=200`、`PATIENCE=25`（Ultralytics 早停；200 为上限）

## 问题

（无）
