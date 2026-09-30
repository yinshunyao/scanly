# 封边缺陷 Go 推理服务（加密 ONNX + GPU License）

产线 HTTP 服务：解密 ONNX、按 NVIDIA GPU 的 uuid/name/serial 三项全绑定校验 License、返回与 Python 版相同的预测 JSON。 **没有 Gradio。**

接口字段见 [`scanly/doc/02-dr/缺陷推理/01.HTTP接口文档.md`](../../doc/02-dr/缺陷推理/01.HTTP接口文档.md)。运行参数见 [`scanly/doc/99-ops/run_go.md`](../../doc/99-ops/run_go.md)。

默认端口 **37871**（Python 版为 37870）。

## 1. 交付物怎么保护

| 文件 | 谁持有 | 说明 |
|:---|:---|:---|
| `*.onnx.enc` + `*.meta.json` | 现场 | 密文与类名元数据 |
| `config/license.json` | 现场 | 绑定 GPU（uuid+name+serial）+ 内含 AES 密钥（已签名） |
| `*.key.json`、`keys/license_ed25519.pem` | **仅厂商** | 模型密钥与签发私钥，禁止随安装包分发 |
| Go 二进制 | 现场 | 内嵌验签公钥；License 通过后才在内存解密 |

换签发密钥后必须重新编译本服务（公钥 `internal/license/license_ed25519.pub`）。

## 2. 导出并加密模型

在 `scanly/train/convert_2_onnx.py` 修改变量后运行（与 `train.py` 相同，不强制命令行）：

- `MODEL_PATH`：训练得到的 `.pt`
- `TASK`：`obb` 或 `detect`
- `OUTPUT_DIR`：默认 `scanly/train/output/onnx/`

产出：`{id}.onnx.enc`、`{id}.meta.json`、`{id}.key.json`。把前两个拷到 `models/`，`.key.json` 留在厂商侧。

依赖：`pip install -r scanly/train/requirements.txt`（含 `cryptography`）。

## 3. 签发 License

```bash
pip install -r scanly/defects/predict_go/tools/requirements.txt
```

1. 现场执行 `nvidia-smi --query-gpu=uuid,name,serial --format=csv,noheader`，记下三列。例如：
   `GPU-aec2d77c-763b-4295-4875-55b9887cd265, Tesla V100S-PCIE-32GB, 1562721015128`
2. 打开 `tools/gen_license.py`：
   - 首次：`INIT_KEYS = True` 运行一次，生成私钥并把公钥写入 Go embed，然后 **重新编译**。
   - `LIST_LOCAL_GPUS = True` 可打印本机 uuid/name/serial。
   - 填写 `GPUS`（可多张卡，每条必须含 uuid、name、serial）、`AES_KEY_PATH`、`EXPIRES_AT`（空=不过期）。
3. 运行脚本，得到 `config/license.json`。

本机 **任一** GPU 的 uuid、name、serial **三项全部** 与列表中某一条一致才通过。只对上 UUID 或只对上型号不算授权。无 NVIDIA GPU 时服务可启动，但 License 不会通过，预测不可用。

## 4. ONNX Runtime 动态库

Go 通过 ONNX Runtime 推理；`ort_lib_path` 为空时按操作系统自动选目录：

| 平台 | 放置目录 | 库文件 |
|:---|:---|:---|
| Windows | `third_party_win/onnxruntime/` | `onnxruntime.dll`（GPU 包另含 `execution_providers_*.dll`） |
| Linux | `third_party/onnxruntime/` | `libonnxruntime.so`（及软链，见运行说明） |
| macOS | `third_party/onnxruntime/` | `libonnxruntime.dylib` |

从 [ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases) 下载。产线 NVIDIA 推理把 `device` 设为 `cuda`。

## 5. 编译与启动

**发布请打两个二进制**（与现场 ORT 主版本配对）：

```bash
cd scanly/defects/predict_go
bash tools/build_dual.sh
# dist/scanly-defects-go           ← ORT ≥1.24
# dist/scanly-defects-go-ort118    ← ORT 1.18.1（如 V100）
```

| 二进制 | `onnxruntime_go` | 配合的 ORT 库 |
|:---|:---|:---|
| `scanly-defects-go` | v1.26（`go.mod`） | ≥1.24 |
| `scanly-defects-go-ort118` | v1.11.0（`go.mod.ort118`） | **1.18.1** |

```bash
./dist/scanly-defects-go-ort118   # 或 ./dist/scanly-defects-go
```

或设置 `SCANLY_PREDICT_GO_ROOT` 指向本目录后再运行二进制。启动日志含 `flavor=ort-new|ort118`。

`config/predict.json` 中 `obb_model_enc_path`、`obb_meta_path`、`license_path`、`scheme_path`（默认同 Python 的 `../predict/config/scheme.json`）。

## 6. HTTP（无页面）

| 方法 | 路径 | 说明 |
|:---|:---|:---|
| GET | `/` | 服务说明 JSON |
| GET | `/health/ready` | 就绪；License 或模型未就绪时 **503** |
| GET | `/v1/license` | 授权状态（不含密钥） |
| GET | `/v1/labels` | 类别对照 |
| GET/PUT/POST | `/v1/scheme` | 方案查询/配置 |
| POST | `/v1/predict`、`/scanly_predict` | 按相机本机路径推理 |

试调：改 `tools/client_example.py` 变量后运行；本机路径图例见 `scanly/defects/test/test_predict_path_api.py`（`POST /v1/predict`，不上传）。

常见失败 `msg`：`许可证无效` / `许可证已过期` / `未检测到NVIDIA GPU` / `本机GPU信息与授权不一致` / `模型解密失败` / `OBB模型未加载` / `读取图片异常`。
