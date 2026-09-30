# 运维发布目录（99-ops）

## 目录用途

存放 scanly 缺陷推理服务与缺陷训练的启动、端口、模型路径等运行说明，以及发布草稿。

## 文档范围

**允许放入：** 启动命令、监听端口、配置文件路径、环境依赖、`release-note-draft.md`。

**不应放入：** 接口字段契约（见 `doc/02-dr/缺陷推理/01.HTTP接口文档.md`）；需求原文。

## 编辑规则

- 运行说明：`run.md`（Python 推理）、`run_go.md`（Go 加密推理：含 Ubuntu/Windows 编译环境、Ubuntu 运行、导出与 License）、`run_go_cuda11.md`（**CUDA 11.8 + cuDNN 8 + ORT 1.18.1** 并排部署，规避 V100 / CUDNN_FE；**§5 TensorRT 8.6 + ORT 1.17.1**）、`train.md`（训练与 ONNX 导出）
- 细小变更先写入 `release-note-draft.md`；收到「发布整理」后再汇总到 `release-note.md`
- 启动参数或目录约定变更时先改本目录再改代码入口注释

## 协作约定

| 相邻目录 | 关系 |
|:---|:---|
| `doc/02-dr/缺陷推理/` | 接口与模块设计 |
| `defects/predict/` | Python 推理启动入口 `serve.py` |
| `defects/predict_go/` | Go 推理启动入口 `cmd/server` |
| `scanly/train/` | 训练启动入口 `train.py` / `train_core.py`；测试集验证 `test.py` / `test_core.py`；导出 `convert_2_onnx.py` |
