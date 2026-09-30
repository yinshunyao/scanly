# 关联文档

| 类型 | 文档路径 |
|:---|:---|
| 原始需求 | `scanly/doc/01-or/【缺陷推理】ONNX导出加密与Go边端GPU授权.md` |
| 开发设计 | `scanly/doc/02-dr/缺陷推理/03.Go端ONNX加密推理与GPU授权.md` |
| 接口契约 | `scanly/doc/02-dr/缺陷推理/01.HTTP接口文档.md` |

# 测试范围

License 验签 / 过期 / GPU 列表；解密时机；HTTP 与 Python 契约对齐；无 Gradio；判定规则与 scheme 一致。

# 测试用例

| ID | 用例名 | 前置 | 步骤 | 预期 | 自动化 |
|:---|:---|:---|:---|:---|:---|
| TC-L01 | 缺 License | 无 license.json | 启动后 GET `/health/ready` | 503，ready=false，msg 含许可 | 否 |
| TC-L02 | 篡改 JSON | 改 gpus 不重签 | 启动 | 许可证无效，不解密 | 否 |
| TC-L03 | 过期 | expires_at 已过去 | 启动 | 许可证已过期 | 否 |
| TC-L04 | GPU 不匹配 | uuid/name/serial 与本机不完全相同 | 启动 | 本机GPU信息与授权不一致 | 否 |
| TC-L05 | GPU 三项命中 | 列表含本机某一卡的 uuid+name+serial | 启动 | `/v1/license` valid=true，模型可加载 | 否 |
| TC-L06 | 无 NVIDIA | 无 nvidia-smi | 启动 | 未检测到NVIDIA GPU | 否 |
| TC-L07 | 签发多卡 | gen_license 填多条 GPU | 本机只插其中一张且三项一致 | 通过 | 否 |
| TC-L08 | 仅 UUID 相同 | uuid 对、name 或 serial 不对 | 启动 | 不通过 | 否 |
| TC-H01 | 无 Gradio | 服务启动 | GET `/` | JSON 说明，不是 Gradio HTML | 否 |
| TC-P01 | 预测契约 | License+模型就绪 | POST `/v1/predict` | 与 Python 相同的 cameras/defects 二维结构 | 否 |
| TC-P02 | 路径别名 | 同上 | `/scanly_predict` | 与 `/v1/predict` 相同 | 否 |
| TC-P03 | FP16 密文推理 | License+`precision=fp16` 模型就绪、device=cuda | POST `/v1/predict` | 正常检出；不因 float16 I/O 报类型错误 | 否 |
| TC-P04 | FP32 密文回归 | 既有 FP32 密文 | POST `/v1/predict` | 行为与改前一致 | 否 |
| TC-P05 | TensorRT FP16 | ORT 含 TensorRT EP、`device=tensorrt`、**FP32** 密文 | 启动服务或跑 perf | 会话首位为 TensorrtExecutionProvider；可完成推理；缓存目录有引擎文件 | 否 |
| TC-P07 | TensorRT 不用 FP16 ONNX | `DEVICE=tensorrt` 且配置了 FP16 密文 | 跑 perf | 默认跳过 FP16 路或关闭额外 TRT FP16；不得因 NaN `round` 崩溃 | 是（手工脚本） |
| TC-P06 | TensorRT 不可用 | `device=tensorrt` 但无 TRT EP | 启动 | 明确失败，不静默降级为仅 CUDA 当成功 | 否 |
| TC-PERF01 | FP32/FP16 性能对比 | 两套密文（或明文）+ 样例图 + GPU ORT | 改 `test_onnx_infer_perf.py` 变量后运行 | 两种精度均可完成 warmup+计时；输出 mean/p50/p95；两者齐全时有加速比 | 是（手工脚本） |
| TC-PERF02 | TensorRT 对比 | 同上 + DEVICE=tensorrt | 运行 perf | 相对 cuda 的 ort/e2e 延迟可对比；首次慢、二次有缓存 | 是（手工脚本） |
| TC-O01 | Linux ORT 目录 | `ort_lib_path` 空 | Linux 启动且 License 通过 | 从 `third_party/onnxruntime/libonnxruntime.so` 加载 | 否 |
| TC-O02 | Windows ORT 目录 | `ort_lib_path` 空 | Windows 启动且 License 通过 | 从 `third_party_win/onnxruntime/onnxruntime.dll` 加载 | 否 |
| TC-O03 | 双二进制配对 | `build_dual.sh` 产物 | ORT 1.18.1 + `*-ort118`；ORT≥1.24 + 默认二进制 | 各自能 Initialize ORT；交叉混用会 API version 失败 | 否 |
| TC-S01 | 方案 | 可无模型但建议有 License | GET/PUT `/v1/scheme` | 合并/落盘行为同 Python | 否 |
| TC-J01 | 判定 | 单测或合成框 | or/and/length_only、面积通道 | 与 scheme.py 一致 | 否 |

不在本迭代强制：与 `.pt` 逐框数值对齐、满速产线、对抗逆向。

# 自动化测试标识

| 层级 | 框架 | 说明 |
|:---|:---|:---|
| 手工 | curl / gen_license.py | License 与 HTTP |
| 性能脚本 | `defects/test/test_onnx_infer_perf.py` | FP32/FP16 同图推理延迟；IDE 改变量运行 |
| 单测 | 默认不写 | 除用户明确要求外不写断言型自动化 |
