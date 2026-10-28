# 发布记录草稿

## 2026-10-26

- 4090 缺陷推理改挂 RT-DETRv2 `3.3.1.onnx`（`GlueSeam` / `Tackless` / `GlueGap`，1024）。`class_names` 顺序与导出一致。该机 `.venv` 的 onnxruntime 带 CUDAExecutionProvider；没有该 Provider 时回退 CPU。
- Python 缺陷推理支持检测 mark 训练检查点 `.pth`：`infer_type=detect`，边长与类名跟检查点（mark 检查点为 `GlueSeam`、`Tackless`、`ResidualTape`，1024）。`ResidualTape` 仍按方案屏蔽。
- 结果图中文标签改用随包字体 `predict/fonts/NotoSansSC-label.ttf`，避免服务器没有中文字体时「胶缝」「标」「排」显示成方框。

## 2026-10-13

- 检测 mark 的 test 画框改到该次 run 目录的 `vis/`（`predictions.jsonl` 含 IoU）。训练结束把 test 指标写到同目录的 `test_metrics.json`。原先写在数据集根的 `output_dir/vis/`，归档 run 时带不走。
- 检测 mark 的 test 统计不再只靠运行库评估。画框使用的同一套匹配写入该次 run 的 `test_summary.json`（总计数、按类 TP/FP/FN、precision/recall、至少中一条的图片数），`vis/vis_summary.json` 与它相同。评估失败时这份文件仍然在。

## 2026-10-12

- 检测 core 按 `data_cfg.json` 的 `val_classes` 选 `best.pth` 和早停：只对这些类做不加权平均。全体 COCO 指标仍写入日志；训后指标增加 `val_select`。空列表保持按全部类别选模。
- 修复 `val_classes` 非空时首轮 val 后崩溃：`params.iouThrs` 按序列读取，不再对 numpy 数组做 `or []`。训练 loss 先 `detach` 再判断是否有限，去掉 requires_grad 标量警告。
- 新增检测 mark 训练入口 `train_mark.py`，配置在 `train_detect_mark/train_config.json`。数据仍用公共 `train_detect_cfg/data_cfg.json`，直接训练已准备的 YOLO 目录。目录、配置和日志不写上游实现名称。
- 检测 mark 的 `imgsz` 定为 1024，与 YOLO / 检测 core 对齐。`size=s` 接受该边长；显存不够再把 `batch` 降到 2、`grad_accum_steps` 提到 2。
- 检测 mark：`model_path` 指定本地预训练权重；公共 `val_classes` 作为 test 评估清单，不写则用 `train_classes`。test 画框写到 `output_dir/vis/`，可单独跑 `test_mark.py`。
- 检测 mark 的运行库写入 `scanly/train/requirements.txt`（`rfdetr[train,loggers]>=1.11.2`）。
- 评估画框的相邻标签先下移再右移，避免 FN/FP 文字叠在一起。检测 core 与检测 mark 共用该画法。
- 检测 core 训练结束的 test 评估默认把画框图写到该次 run 目录的 `vis/`（`save_vis` / `vis_conf` / `vis_iou`）。val 不画。
- 检测 core 与检测 mark 的 test 评估增加推理时延：core 写入 `test_metrics.json` 的 `latency`，mark 写入 `vis/vis_summary.json`。多次调用时不含第一次。

## 2026-10-10

- 胶缝/开胶 core 本轮：`single_cls=false`，`train_classes` 增加 `ResidualTape`。test/val 指标增加 `per_class_recall50`，日志按类打印 AP50、AP、IoU=0.5 召回。

- `convert_from_ls.py`：`rotation ≠ 0` 时按 Label Studio 存法绕矩形左上角顺时针旋转，取上下两端最大范围写成轴对齐框，不再忽略旋转。
- `data_cfg.json` 的 `voc_xml_path` 支持字符串或目录列表。多目录时 `voc_split.split_each_dir` 默认打开：每个目录单独按 val/test 比例划分再合并；关闭则全部打乱后一次划分。

## 2026-10-08

- 检测 core 换骨干只改 `train_detect_core/train_config.json` 的 `backbone`（`r18` / `r34` / `r50`）。训练 yaml 自动用 `configs/core/core_*vd_scanly.yml`，不再手写 `model_yml`。

## 2026-09-30

- `test_core.py` 增加评估可视化开关 `SAVE_VIS`（入口默认开）：写出 `vis/` 画框图（TP/FP/FN + 置信度）、`predictions.jsonl`、`vis_summary.json`；`VIS_CONF`/`VIS_IOU` 不影响 mAP。实现自包含于 `scanly/train/test_core_vis.py`，不依赖 `defects/`。

- 修复检测 core ONNX 评估：`test_core` 对 `.onnx` 强制 batch=1（导出图内 MHA/Reshape 写死 batch=1，训练 batch=4 会触发 `node_view` Reshape 失败）。新增 `export_core.py` 单独导出；勿与 YOLO `tools/convert_2_onnx.py` 混用。

- `train_core.py` 训后顺序改为：最优 `.pth` 做 val →（可选）导出 ONNX → test 优先用 ONNX（`export_onnx=false` 时回退 `.pth`）。

- `prepare_dataset.py` 迁入 `train/train_detect_cfg/`；入口改为 `from train_detect_cfg.prepare_dataset import ...`，便于 `train_scanly` 加密包自包含数据准备。

- 检测 core 路径脱敏：`rtdetrv2_pytorch`→`detect_core`，`train_rtdetrv2.py`→`train_core.py`，`test_rtdetrv2.py`→`test_core.py`；Scanly yaml 迁到 `configs/core/core_*_scanly.yml`。README 补充 pyfernet 加密/运行命令。

- 训练配置解耦：公共数据 `train/train_detect_cfg/`，YOLO / RT-DETRv2 / OBB 超参分别在 `train_detect_yolo` / `train_detect_core` / `train_detect_obb`；入口脚本读 JSON，不再在 `__main__` 堆数据列表。
- `convert_from_ls.py` 入口改为仅配置输入目录 + 输出目录；默认递归子目录收集 JSON/图像；LS 图名 `_` 可匹配本地空格文件名。

## 2026-09-22

- 新增 `defects/predict/predict_dir.py`：目录批量离线推理，输出画框图（`vis/`）与 Pascal VOC（`JPEGImages` + `Annotations`）。
- Python 推理 `defects/predict` 支持明文 `.onnx`：YOLO 导出走 Ultralytics；RT-DETRv2 导出走 onnxruntime（`imgsz` / `class_names`）。`predict.json` 增加 `_readme` 说明 `infer_type` 与路径配对；依赖补充 `onnxruntime`。
- `train_core.py` 默认骨干改为 RT-DETRv2-S（R18）；R34 / R50 配置保留为注释，改三行 `MODEL_YML`/`TUNING`/`TUNING_URL` 即可切换。`run_prefix` 默认 `core_r18`。

## 2026-09-21

- `train_core.py`：`fit` 结束后自动对 `best.pth`（无则 `last.pth`）先做 val，再在 `TEST_RATIO>0` 且 test 有图时做 test；指标写到该次 run 的 `val_metrics.json` / `test_metrics.json`。

## 2026-09-14

- 新增 `scanly/train/test_core.py`：对 RT-DETRv2 训练产出的 `.pth` / `.onnx` 做 test 划分 COCO mAP，写出 `test_metrics.json`。评估辅助函数放在测试脚本内，不依赖训练入口新增符号（现场只同步该文件即可 import）。写指标时按序列读取 COCO `stats`（numpy 数组不可用 `or []`）。
- `train_core.py` 按现场 `prepare_dataset` 签名传参，兼容无 `test_ratio` 的旧副本。
- 新增 `scanly/train/train_core.py`：RT-DETRv2 PyTorch 检测训练，数据配置对齐 `train.py`；实现自包含于 `scanly/train/detect_core/`。

## 2026-09-11

- `train.py` 默认 `EPOCHS=200`、`PATIENCE=25`（Ultralytics 早停；200 为上限）。
- 整理 v1.1–v2.1 训练日志说明：`scanly/doc/测试结果/各版本训练日志分析.md`（磁盘 `训练结果/说明.md`）。v2.1 为 cam2 四类 detect；不宜与 v1.5 单类 / v1.3 OBB 直接比 mAP。
- `train.py` 去掉 `RUN_NAME`：不再覆盖同名 run；Ultralytics 在 `runs/` 下自动递增 `train` / `train2` / …。
- `train.py` / `prepare_dataset.py` 支持扁平 VOC 源：`VOC_XML_PATH`（`JPEGImages` + `Annotations`）转为 YOLO 后训练，不得与 `OUTPUT_DIR` 同路径。

## 2026-09-06

- 新增 `scanly/train/tools/split_voc_by_camera.py`：按源 `cam-all-0901` 的 `cam1&3`/`cam2` stem 将扁平 VOC 拆到 `样本数据/came-split-0906-voc/`，供分相机组训练。
- 标注口径补充：`BandBroken`=确定缺陷，`BandBroken2`=疑似缺陷（常见伴随：可见破损面 vs 看不到破损面）；角缺料仍优先 `GlueGap`。见对照表与类别角色文档。

## 2026-09-02

- `convert.py` 支持多层 YOLO 导出目录（如 `cam-all-0901/cam/{缺陷}/images|labels`）转 Pascal VOC；输出扁平到 `JPEGImages/` 与 `Annotations/`，同名同内容去重。
- 新增 `scanly/train/test.py`：配置对齐 `train.py`，默认对 `data.yaml` 的 test 划分做 Ultralytics val，写出 `test_metrics.json`。
- `train.py` 增加 `SINGLE_CLS`（默认 `False`）：开启后传 Ultralytics `single_cls`，已纳入的全部框按单类训练，不改写预处理标签。
- `train.py` 的 `TRAIN_CLASSES` 改为显式 18 类，按缺陷 / 非缺陷分组注释；不含 `Above`。
- 记录封边类别角色：12 类核心缺陷；`ResidualTape`/`Hole`/`Zigzag`/`Label` 为屏蔽非缺陷（训练可保留，推理默认不报出）。见 `doc/01-or/【缺陷类别】核心缺陷与屏蔽非缺陷.md`。
- `/v1/labels` 对照：`Below`/`Shorter`/`BandBroken`/`Scratch` 的 `in_train_set` 改为 true，与当前 `TRAIN_CLASSES` 统计一致。

## 2026-08-23

- `run_go.md` TensorRT 系统库安装：必须钉 `*+cuda11.8`（或直接装 local repo deb）并 `apt-mark hold`。不钉版本时会被 CUDA 网络源的 `libnvinfer10 10.16+cuda13.2` 抢走。
- `run_go.md`：V100（SM 70）不支持 TensorRT 10；`Target GPU SM 70 is not supported` 时保持 `device=cuda`，无需重导 ONNX。
- `run_go_cuda11.md` **§5**：V100 TensorRT 适配手册（卸 TRT 10、钉装 **8.6 cuda-11.8**、Python 改 **ORT 1.17.1**、现有 ONNX 不重导；Go ort118 仍走 CUDA EP）。`env_cuda11.sh` 支持 `TENSORRT86_LIB`。
- TensorRT 半精度：用 FP32 ONNX + EP `trt_fp16`；FP16 ONNX 再叠 TRT FP16 会 NaN。解码跳过非有限框；perf 在 `DEVICE=tensorrt` 时默认跳过 FP16 路。

## 2026-08-21

- `predict.json` 支持 `device=tensorrt`（`trt_fp16_enable` / `trt_engine_cache_path`）；Python `server_onnx` 与 Go 引擎启用 TensorRT EP + FP16；性能脚本可切 `DEVICE=tensorrt`。
- 模型目录约定：全精度 `predict_go/models/v1.3-onnx/`，半精度 `predict_go/models/v1.3-onnx-fp16/`；`predict.json` / 性能脚本 / `convert_2_onnx.py` 输出路径已对齐。
- `convert_2_onnx.py`：GPU/FP16 导出前默认清空本进程 `LD_LIBRARY_PATH`（`SANITIZE_LD_LIBRARY_PATH`），规避与 Go ORT CUDA11/cuDNN 冲突导致的 `GET was unable to find an engine`。
- `doc/99-ops/run_go_cuda11.md` §3.3 / 故障表补充：推理 `env_cuda11.sh` 与 Ultralytics 导出环境隔离操作。
- `convert_2_onnx.py` 支持半精度导出：`HALF=True`（默认）走 Ultralytics FP16 ONNX，须 `DEVICE=cuda/0`；meta 增加 `precision` / `half`。
- Go `predict_go` 与 Python `server_onnx.py` 按 ONNX 输入输出 dtype 兼容 FP16 / FP32 密文推理。
- 新增 `defects/test/test_onnx_infer_perf.py`：同图对比 FP32/FP16 推理延迟（e2e + session.run）。
- 缺陷推理 Python 服务新增 `POST /v1/predict/upload`：multipart 上传图像推理，响应与路径接口一致；`POST /v1/predict` 保持本机路径不变。
- 接口/测试设计已同步；`client_example.py` 支持 `MODE=upload`。
- `doc/99-ops/run_go.md` 补充 Ubuntu / Windows 编译用 Go 环境准备（Go 1.24、CGO、gcc/MinGW、ORT 动态库）。
- `run_go.md` 补充 Ubuntu 转 ONNX/加密操作，以及本机 License 签发步骤与交付清单。
- `scanly/train/requirements.txt` 补充导出 ONNX 依赖：`onnx`、`onnxslim`、`onnxruntime`。
- `convert_2_onnx.py` 默认保留明文 `.onnx`（`KEEP_PLAIN_ONNX=True`），由人工删除后再交付。
- `run_go.md` 补充 Ubuntu 运行操作说明（交付目录、ORT、配置、启动与自检；运行无需 Go 工具链）。
- `run_go.md` 补充国内 `GOPROXY`（如 goproxy.cn）以解决 `proxy.golang.org` 超时。
- `run_go.md` 补充 Linux ORT：`.so` / `.so.1` 为软链接、上传易丢，服务器上 `ln -sf` 重建或改 `ort_lib_path`。
- Go 默认 ORT：Linux `third_party/onnxruntime/`，Windows `third_party_win/onnxruntime/`（与 Linux 隔离）。
- `run_go.md` 补充 GPU/ORT/CUDA/cuDNN 版本对齐与 V100S（sm_70）选包注意（V100 推荐 ORT **1.18.1**）。
- 方案 B：`tools/build_dual.sh` 发布 `scanly-defects-go`（ORT≥1.24）与 `scanly-defects-go-ort118`（ORT **1.18.1**，`onnxruntime_go v1.11.0` / API 18）。原 ort123/1.23.2 线已切换为 ort118（Python 1.18.1 GPU 已在 V100 验证通过）。
- 新增 `predict_go/server_onnx.py`：Python 冒烟服务，读同一套 `config/predict.json`/License/密文模型；用 **pip onnxruntime(-gpu)** 推理（不加载 third_party C 库），默认端口 **37872**。
- 新增 `doc/99-ops/run_go_cuda11.md`：并排部署 **CUDA 11.8 + cuDNN 8 + ORT 1.18.1**（解决 `libcublasLt.so.11` 缺失；规避 V100 上 ORT≥1.20 CUDNN_FE）。

## 2026-08-20

- 训练目录新增 `convert_2_onnx.py`：YOLO `.pt` 导出 ONNX 并 AES-GCM 加密。
- 新增 Go 推理服务 `scanly/defects/predict_go`：内存解密 ONNX、HTTP 与 Python 版同契约（无 Gradio）、NVIDIA GPU License（uuid+name+serial 三项全绑定）。
- License 由 `predict_go/tools/gen_license.py` 签发；运行说明见 `scanly/doc/99-ops/run_go.md`。
- 训练默认数据切到合并批次 `scanly/样本数据/cam123-0820`（相机 1/2/3）；默认 18 类，不含 `Above`。
- 检测 / OBB 输出目录与 run 名改为 `cam123-0820` / `cam123-0820-obb`。

## 2026-08-19

- 新增封边缺陷 YOLO 检测训练入口（`scanly/train`）：指定原始目录与训练类别，按类名过滤，多边形标签转为检测框后训练。
- 新增 OBB 训练入口 `scanly/train/train_obb.py`：保留四点标签，使用 YOLO11-OBB 训练。
- 推理服务支持 OBB：`predict.json` 增加 `infer_type` / `obb_model_path`（`infer_type` 可选，默认 `obb`）；`POST /v1/predict` 可传 `infer_type`；结果统一含 `obb.points`（detect bbox 转 angle=0 四点），判定按旋转框宽高。
- 修复 Gradio 结果图中文标签乱码：扩展系统 CJK 字体候选并对 TTC 多 index 校验；Ubuntu 补充文泉驿/Noto 路径。
- Gradio 下方 JSON 拆成「本次检出结果」与「判定方案配置」，避免把方案 defects 误当成检出列表。
- 默认适配 `scanly/样本数据/cam2-0818`；运行说明见 `scanly/doc/99-ops/train.md`。

## 2026-08-15

- 新增封边缺陷推理服务第一版（`scanly/defects/predict`）：HTTP JSON（本机图像路径）+ Gradio 调试页。
- 默认方案对齐客户「缺陷管理 / 精度」页（方案名 `20260605`）。
- Swagger `/docs` 声明推理请求/响应模型，200 示例可见按相机分组的缺陷结构。
- 方案 `channels` 按精度页拆成 `ng_area` / `label_area` / `discharge_area`，不再用全局 `area`。
