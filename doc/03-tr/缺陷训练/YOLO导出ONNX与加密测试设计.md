# 关联文档

| 类型 | 文档路径 |
|:---|:---|
| 原始需求 | `scanly/doc/01-or/【缺陷推理】ONNX导出加密与Go边端GPU授权.md` |
| 开发设计 | `scanly/doc/02-dr/缺陷训练/02.YOLO导出ONNX与加密.md` |

# 测试范围

导出路径解析、detect/obb 任务、加密文件头、无密钥则生成、复用已有密钥、可选保留明文、半精度（FP16）导出与 meta 精度字段。

# 测试用例

| ID | 用例名 | 前置 | 步骤 | 预期 | 自动化 |
|:---|:---|:---|:---|:---|:---|
| TC-E01 | 缺 pt | 路径不存在 | 运行 convert | 失败并提示文件不存在 | 否 |
| TC-E02 | OBB 导出加密 | 有效 obb `.pt` | TASK=obb 运行 | 产出 `.onnx.enc`、`.meta.json`、`.key.json` | 否 |
| TC-E03 | 默认保留明文 | KEEP_PLAIN_ONNX=true（默认） | 导出成功 | 交付目录与 Ultralytics 导出路径仍有明文 `.onnx`；密文/meta/key 正常 | 否 |
| TC-E03b | 自动删明文 | KEEP_PLAIN_ONNX=false | 导出成功 | 输出目录无明文 `.onnx` 或已删除 | 否 |
| TC-E04 | 复用密钥 | 已有 `.key.json` | AES_KEY_PATH 指向它再转另一模型 | 两密文可用同一 License 密钥 | 否 |
| TC-E05 | meta names | 导出成功 | 读 meta.json | 含 task、imgsz、names | 否 |
| TC-E06 | FP16 导出 | GPU 可用、HALF=true、DEVICE=cuda/0 | 导出成功 | meta `precision=fp16`、`half=true`；密文可被 Go/Python 加载推理 | 否 |
| TC-E07 | HALF+CPU 拒绝 | HALF=true、DEVICE=cpu | 运行 convert | 失败并提示半精度需 GPU | 否 |
| TC-E08 | FP32 导出 | HALF=false | 导出成功 | meta `precision=fp32`；与既有推理兼容 | 否 |
| TC-E09 | LD_LIBRARY_PATH 冲突 | shell 已设 CUDA11/cuDNN 路径、HALF=true | 默认 SANITIZE 开启后导出 | 不再报 `GET was unable to find an engine`；或手工 `LD_LIBRARY_PATH=` 后成功 | 否 |

# 自动化测试标识

默认本迭代不写自动化代码。手工：改 `convert_2_onnx.py` 变量后 IDE 运行。
