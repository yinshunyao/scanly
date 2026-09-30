# Go / Python 加密 ONNX 推理 — CUDA 11 + cuDNN 8 + ORT 1.18.1

> 配套总册：[`run_go.md`](./run_go.md)（通用编译、License、HTTP、双二进制）。  
> 本手册解决两件事：  
> 1. **CUDA EP**：现场已有 CUDA 12 / cuDNN 9，要用 ORT 1.18.1（CUDA 11 构建）跑 GPU。  
> 2. **TensorRT EP（V100 / SM 70）**：现有 ONNX 按 CUDA 11 导出，须配 **TensorRT 8.6**（见 **§5**）。**不要**用 TensorRT 10，也**不必重导 ONNX**。

## 背景与目标

在 Tesla V100 等卡上，ORT ≥1.20（含 1.23）走 **cuDNN Frontend**，易出现：

`CUDNN_FE` / `no kernel image is available for execution on the device`

验证结论：Python `onnxruntime-gpu==1.18.1` 与 Go 同错 → **不是 Go 绑定问题**。  
规避路线：改用 **ORT 1.18.1（CUDA 11 + cuDNN 8，无 FE Conv）**。

本手册目标：

1. **不必卸载** 现有 CUDA 12 / 驱动 / cuDNN 9  
2. **并排安装** CUDA 11.8 Toolkit + cuDNN 8（apt：`libcudnn8`，与 cuDNN 9 共存）  
3. 让进程能找到 `libcublasLt.so.11`、`libcudnn.so.8`  
4. Python 冒烟 `server_onnx.py` +（可选）Go + 下载的 ORT 1.18.1 GPU 包  
5. （可选）V100 上启用 TensorRT：并排 **TensorRT 8.6（CUDA 11.8）** + Python **ORT 1.17.1**（与 1.18.1 / TRT 10 不是同一条线）

## 版本对齐（强制）

| 组件 | 版本 | 说明 |
|:---|:---|:---|
| NVIDIA 驱动 | 保持现网（如 580） | 新驱动可跑旧 CUDA 用户态库 |
| CUDA Toolkit（并排） | **11.8** | 提供 `libcublasLt.so.11` 等 |
| cuDNN（并排） | **8.x**（apt：`libcudnn8`，如 8.9.7） | 与 cuDNN 9 共存；ORT 1.18 用 `libcudnn.so.8`，勿改全局 `libcudnn.so`→9 |
| Python ORT | **`onnxruntime-gpu==1.18.1`** | PyPI 上该版本默认是 **CUDA 11** 构建 |
| Go 用动态库 | **`onnxruntime-linux-x64-gpu-1.18.1.tgz`** | 注意：不要下成 `*-gpu-cuda12-1.18.1.tgz` |
| Go 绑定（若跑 Go） | `onnxruntime_go` **v1.11.0**（API **18**） | 与 ORT 1.18.x 配对 |
| TensorRT（仅 `device=tensorrt`） | 见 **§5** | **V100 必须 8.6** + ORT **1.17.1**（`libnvinfer.so.8`）。ORT **1.18.1** 官方配 TensorRT **10**（`libnvinfer.so.10`），V100 会报 `Target GPU SM 70 is not supported` |

### 你当前报错的含义

```text
Failed to load library libonnxruntime_providers_cuda.so with error:
libcublasLt.so.11: cannot open shared object file: No such file or directory
→ session providers=['CPUExecutionProvider']
```

说明：已装上 **CUDA 11 版** 的 ORT 1.18.1，但系统里只有 CUDA 12 的 `libcublasLt.so.12`，**没有 `.so.11`**。  
处理：装 CUDA 11.8 用户态库，并把其 `lib64` 放进 `LD_LIBRARY_PATH`（见下文），**不是**再装一份 ORT。

---

## 0. 先确认现状

```bash
nvidia-smi
nvcc --version 2>/dev/null || true
ls -l /usr/local/cuda* 2>/dev/null
ldconfig -p | grep -E 'cublasLt|cudnn' | head -40
python3 -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())" 2>/dev/null || true
```

期望最终（本节完成后）至少能看到：

```text
libcublasLt.so.11 => /usr/local/cuda-11.8/...
libcudnn.so.8     => /usr/lib/x86_64-linux-gnu/...   # apt 安装 libcudnn8
```

---

## 1. 并排安装 CUDA Toolkit 11.8（保留 CUDA 12）

**不要** `apt remove` 掉 CUDA 12。两套 Toolkit 可共存：`/usr/local/cuda-11.8` 与 `/usr/local/cuda-12.x`。

### 1.1 用 NVIDIA 官方 deb（Ubuntu 推荐）

按本机 Ubuntu 版本选包（示例 **22.04 / jammy**；20.04 用 `ubuntu2004`）：

```bash
# 若尚未配置 cuda-keyring，参考：
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# 只装 11.8 toolkit（不必装旧驱动）
sudo apt-get -y install cuda-toolkit-11-8
```

装完检查：

```bash
ls /usr/local/cuda-11.8/lib64/libcublasLt.so.11
/usr/local/cuda-11.8/bin/nvcc --version
```

国内下载慢时：可从镜像站取同名 `cuda-repo` / 本地 offline runfile，或在有网机器下载 `cuda_11.8.*_linux.run` 后拷到现场：

```bash
# 官方 runfile 示例（版本号以 NVIDIA 下载页为准）
sudo sh cuda_11.8.0_520.61.05_linux.run --toolkit --silent --override
# 确认出现 /usr/local/cuda-11.8
```

### 1.2 不要改默认 `/usr/local/cuda` 软链（可选建议）

若现网其它软件依赖 CUDA 12，**保持** `cuda → cuda-12.x`。  
ORT 1.18 只通过 **`LD_LIBRARY_PATH` 优先找 11.8**，不必改全局默认。

---

## 2. 并排安装 cuDNN 8（与 cuDNN 9 共存）

ORT 1.18.1（CUDA 11 包）需要 **`libcudnn.so.8`**，与现网 **cuDNN 9** 可并存，**不要卸载** `libcudnn9-*`。

### 2.1 apt 安装（Ubuntu + NVIDIA 源，推荐）

现场已验证（Ubuntu 22.04 + `developer.download.nvidia.cn`）：

```bash
sudo apt-get update
apt-cache search libcudnn8 | head -20

sudo apt-get -y install libcudnn8
# 包版本可能显示为 8.9.7.29-1+cuda12.2（标签带 cuda12），
# 仍提供 libcudnn.so.8，可供 ORT 1.18.1 使用。
```

#### 安装位置与核对

```bash
dpkg -L libcudnn8 | grep -E 'libcudnn\.so'
# 期望类似：
#   /usr/lib/x86_64-linux-gnu/libcudnn.so.8
#   /usr/lib/x86_64-linux-gnu/libcudnn.so.8.9.7

ls -l /usr/lib/x86_64-linux-gnu/libcudnn.so*
```

现场并存时常见形态：

```text
libcudnn.so     -> libcudnn.so.9          # 无后缀默认仍指向 9（给 cuDNN9 用）
libcudnn.so.8   -> libcudnn.so.8.9.7      # ORT 1.18 实际会链 .so.8
libcudnn.so.9   -> libcudnn.so.9.25.0
```

要点：

1. **路径是** `/usr/lib/x86_64-linux-gnu/`，**不是** `/usr/local/cudnn-8.9-cuda11/`（后者仅 tar 解压示例）。
2. **`libcudnn.so` 指向 `.so.9` 没关系**，一般**不要**改成指向 `.so.8`（会破坏依赖 cuDNN 9 的软件）。ORT 1.18 通常直接依赖 `libcudnn.so.8`。
3. `ldconfig -p | grep cudnn` 可能大量列出 **cuDNN 9** 子库（`libcudnn_ops.so.9` 等）；只要 `ls` 能看到 `libcudnn.so.8` 即可。

若缺开发头文件或链接报缺子库，再装：

```bash
sudo apt-get -y install libcudnn8-dev
# 仍缺库时：
apt-cache search libcudnn8
```

### 2.2 官网 tar（apt 无 libcudnn8 时）

1. [cuDNN Archive](https://developer.nvidia.com/cudnn) 选 **cuDNN 8.9.x for CUDA 11.x，Linux x86_64，tar**。
2. 解压到独立目录（勿覆盖 cuDNN 9）：

```bash
tar -xvf cudnn-linux-x86_64-8.9.*_cuda11*-archive.tar.xz
sudo mkdir -p /usr/local/cudnn-8.9-cuda11
sudo cp -a cudnn-linux-x86_64-*-archive/include /usr/local/cudnn-8.9-cuda11/
sudo cp -a cudnn-linux-x86_64-*-archive/lib     /usr/local/cudnn-8.9-cuda11/
ls /usr/local/cudnn-8.9-cuda11/lib/libcudnn.so.8*
# 若目录名是 lib64，把下面环境变量改成 lib64
```

---

## 3. 环境变量（每次跑 GPU 推理前）

**顺序很重要**：`cuda-11.8` 的 `lib64` 必须能被找到（解决 `libcublasLt.so.11`）。  
cuDNN 用 apt 时，系统目录已在默认搜索路径；仍建议显式写入 `LD_LIBRARY_PATH`，并保证 **CUDA 11.8 在 CUDA 12 之前**。

### 3.1 apt 安装 cuDNN 8 时（推荐）

```bash
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/lib64:/usr/local/cuda-11.8/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}

# 自检
ls /usr/local/cuda-11.8/lib64/libcublasLt.so.11
ls -l /usr/lib/x86_64-linux-gnu/libcudnn.so.8

ORT_CUDA=$(python3 -c 'import onnxruntime, pathlib; print(pathlib.Path(onnxruntime.__file__).parent / "capi" / "libonnxruntime_providers_cuda.so")')
echo "$ORT_CUDA"
ldd "$ORT_CUDA" | grep -E 'cublasLt|cudnn'
# 期望看到 libcublasLt.so.11 与 libcudnn.so.8（不要只有 .so.12 / .so.9）
```

仓库脚本（apt 默认路径已写好）：

```bash
cd ~/scanly/predict_go
source ./env_cuda11.sh
```

### 3.2 tar 安装 cuDNN 8 时

```bash
export CUDNN8_LIB=/usr/local/cudnn-8.9-cuda11/lib   # 或 lib64
source ./env_cuda11.sh
```

### 3.3 同一机做 YOLO→ONNX 导出时（必读）

`source env_cuda11.sh` / 上文 `LD_LIBRARY_PATH` **只给 ORT 1.18.1 推理用**。  
同一 shell 里再跑 Ultralytics / PyTorch（如 `torch+cu126`）导出时，系统 **CUDA 11.8 + cuDNN 8** 会盖住 PyTorch wheel 自带库，FP16 GPU 导出常报：

```text
RuntimeError: GET was unable to find an engine to execute this computation
```

（出现在 `convert_2_onnx.py` → `model.export(...)` 的 dry-run `Conv2d`，与 V100 是否支持 FP16 无关。）

**正确做法（任选）：**

```bash
# 方式 A：新开终端，不要 source env_cuda11.sh；用仓库脚本默认行为
cd ~/scanly/train   # 或仓库 scanly/train
# convert_2_onnx.py 默认 SANITIZE_LD_LIBRARY_PATH=True，会清空「本进程」LD_LIBRARY_PATH 再导出
python3 convert_2_onnx.py

# 方式 B：当前 shell 已 source 过 env_cuda11.sh 时，强制清空后再导出
LD_LIBRARY_PATH= python3 convert_2_onnx.py

# 方式 C：不需要 FP16 时
# 编辑 convert_2_onnx.py：HALF=False，再运行（推理侧已兼容 FP32）
```

| 场景 | `LD_LIBRARY_PATH` |
|:---|:---|
| Go / `server_onnx.py` GPU 推理 | **需要** `env_cuda11.sh`（含 cuda-11.8 + cuDNN8） |
| `convert_2_onnx.py` FP16 导出 | **不要**沿用上述路径；清空或交给脚本 sanitize |

更细说明见 `run_go.md`「转 ONNX 并加密」、`doc/91-qa/【缺陷训练】ONNX半精度导出GET引擎错误.md`。

---

## 4. Python：安装 ORT 1.18.1 并冒烟

### 4.1 国内镜像（阿里云）

```bash
pip uninstall -y onnxruntime onnxruntime-gpu

pip install onnxruntime-gpu==1.18.1 \
  -i https://mirrors.aliyun.com/pypi/simple/ \
  --trusted-host mirrors.aliyun.com

# 其它依赖（冒烟服务）
pip install cryptography pillow numpy \
  -i https://mirrors.aliyun.com/pypi/simple/ \
  --trusted-host mirrors.aliyun.com
```

备选清华：`-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn`。

确认版本：

```bash
python3 -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
# 期望：1.18.1 且列表含 CUDAExecutionProvider
```

### 4.2 配置与启动

`config/predict.json`：

```json
"device": "cuda"
```

```bash
cd ~/scanly/predict_go    # 或仓库 scanly/defects/predict_go
source ./env_cuda11.sh    # 上一节写的环境脚本
python3 server_onnx.py    # 默认 37872
```

**成功标志：**

```text
[ort] version=1.18.1 available=[..., 'CUDAExecutionProvider', ...]
[ort] session providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
```

**失败仍见 `libcublasLt.so.11`：** `LD_LIBRARY_PATH` 未生效或未装 11.8（回头做 §1 / §3）。

**失败仍见 `libcudnn.so.8` / cuDNN：** 未装 cuDNN 8 或路径未排在前面（§2 / §3）。

**V100 上要 TensorRT：** 不要改 `device=tensorrt` 配 ORT 1.18.1 / TensorRT 10。按 **§5** 换成 TensorRT 8.6 + ORT 1.17.1。现有 CUDA11 密文 ONNX **不用重导**。

用客户端测（改 `BASE_URL` 为 `http://<host>:37872`）：

```bash
python3 ../test/test_predict_path_api.py
# 或仓库内：scanly/defects/test/test_predict_path_api.py
```

Python 冒烟 **通过** 后，再部署同代 Go 库才有意义。CUDA EP 即到此为止。V100 上要 TensorRT，继续 **§5**（会换成 ORT 1.17.1，与本节 1.18.1 **不要混装在同一 Python 环境**）。

---

## 5. TensorRT 8.6（V100 / SM 70）

现有 ONNX 已按 CUDA 11 导出，**不要重导**。V100 要用 TensorRT，必须换 **8.6**，不能用 10.x。

### 5.1 为什么不能沿用 ORT 1.18.1

| 组合 | V100 上结果 |
|:---|:---|
| ORT **1.18.1** + TensorRT **10.x**（`libnvinfer.so.10`） | 建引擎失败：`Target GPU SM 70 is not supported` |
| ORT **1.18.1** + TensorRT **8.6**（`libnvinfer.so.8`） | TensorRT EP 加载失败：1.18 官方配 TRT 10，找的是 **`.so.10`** |
| ORT **1.17.1** + TensorRT **8.6**（`libnvinfer.so.8`） | **本手册 TensorRT 线**（ORT 1.17 官方配 TRT 8.6，支持 SM 70） |
| ORT **1.18.1** + `device=cuda` | 仍可用（§1–4），只是没有 TensorRT |

官方对应关系（节选）：ORT **1.17 → TensorRT 8.6**；ORT **1.18 → TensorRT 10.0**。  
两条线共用：**CUDA 11.8 Toolkit + cuDNN 8 + 现有密文 ONNX**。差别只在 ORT 小版本与 TensorRT 大版本。

Python TensorRT 线请 **另开 venv**（或先卸 1.18.1 再装 1.17.1），避免两个 `onnxruntime-gpu` 互相覆盖。

### 5.2 先卸 TensorRT 10（若已装）

本机若刚装过 `libnvinfer10`（10.9 / 10.16 等），必须先去掉，否则 1.17.1 可能仍链到 `.so.10`，或 apt 把 8.6 依赖解错。

```bash
dpkg -l | grep -E 'nvinfer|nvonnxparsers|nvparsers|tensorrt'
# 若有 libnvinfer10 / libnvinfer-plugin10 / libnvonnxparsers10：
sudo apt-mark unhold libnvinfer10 libnvinfer-plugin10 libnvinfer-vc-plugin10 libnvonnxparsers10 2>/dev/null || true
sudo apt-get remove -y \
  'libnvinfer10' 'libnvinfer-plugin10' 'libnvinfer-vc-plugin10' 'libnvonnxparsers10' \
  'libnvinfer-dev' 'libnvonnxparsers-dev' 'tensorrt' 2>/dev/null || true
# 可选：去掉 TensorRT 10 的 local repo
# sudo apt-get remove -y 'nv-tensorrt-local-repo-*-10.*' || true
sudo apt-get update
ldconfig -p | grep nvinfer || true
# 此时不应再看到 libnvinfer.so.10
```

### 5.3 安装 TensorRT 8.6（必须 CUDA 11.8 变体）

打开 [TensorRT 8.x Archive](https://developer.nvidia.com/nvidia-tensorrt-8x-download)（需登录），选：

- TensorRT **8.6.1**
- Ubuntu **22.04**（20.04 则用 `ubuntu2004`）
- **CUDA 11.8** 的 *TensorRT 8.6 GA for Ubuntu … local repo Deb*

包名示例：`nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-11.8_1.0-1_amd64.deb`  
**不要**下 CUDA 12.0 的 8.6 包。

#### 方式 A：local repo deb（推荐）

与装 TensorRT 10 时相同：须 **钉 `*+cuda11.8`**。CUDA 网络源会把更高版本 / 别的 CUDA 后缀抢走。

```bash
sudo dpkg -i nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-11.8_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-11.8/*-keyring.gpg /usr/share/keyrings/
sudo apt-get update

apt-cache madison libnvinfer8
# 候选里应有 8.6.1.*+cuda11.8

sudo apt-get install -y \
  'libnvinfer8=*+cuda11.8' \
  'libnvinfer-plugin8=*+cuda11.8' \
  'libnvparsers8=*+cuda11.8' \
  'libnvonnxparsers8=*+cuda11.8'
sudo apt-mark hold \
  libnvinfer8 libnvinfer-plugin8 libnvparsers8 libnvonnxparsers8
```

通配仍指向错误源时，直接装 local 目录里的 deb：

```bash
REPO=/var/nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-11.8
sudo apt-get install -y \
  "$REPO"/libnvinfer8_*+cuda11.8_amd64.deb \
  "$REPO"/libnvinfer-plugin8_*+cuda11.8_amd64.deb \
  "$REPO"/libnvparsers8_*+cuda11.8_amd64.deb \
  "$REPO"/libnvonnxparsers8_*+cuda11.8_amd64.deb
sudo apt-mark hold \
  libnvinfer8 libnvinfer-plugin8 libnvparsers8 libnvonnxparsers8
```

ORT TensorRT EP 只需要上述 **runtime `.so.8`**，不必装 `python3-libnvinfer` / 完整 `tensorrt` 元包。

#### 方式 B：tar（免 root / 与 10.x 并排时）

同一下载页取 `TensorRT-8.6.1.*Linux.x86_64-gnu.cuda-11.8*.tar.gz`：

```bash
sudo mkdir -p /usr/local/TensorRT-8.6.1
sudo tar -xzf TensorRT-8.6.1.*.Linux.x86_64-gnu.cuda-11.8*.tar.gz \
  -C /usr/local/TensorRT-8.6.1 --strip-components=1
ls /usr/local/TensorRT-8.6.1/lib/libnvinfer.so.8
# 每次推理前：
export TENSORRT86_LIB=/usr/local/TensorRT-8.6.1/lib
```

### 5.4 核对库与环境

```bash
dpkg -l | grep -E 'libnvinfer8|libnvinfer-plugin8|libnvonnxparsers8|libnvparsers8'
ldconfig -p | grep nvinfer
# 必须有 libnvinfer.so.8 ；不应再有 libnvinfer.so.10

cd ~/scanly/predict_go    # 或仓库 defects/predict_go
export TENSORRT86_LIB=/usr/lib/x86_64-linux-gnu   # tar 则改成 /usr/local/TensorRT-8.6.1/lib
source ./env_cuda11.sh

# 确认 ORT 即将加载的是 .so.8
ls "$TENSORRT86_LIB"/libnvinfer.so.8
```

`env_cuda11.sh` 会把 `TENSORRT86_LIB`（若存在 `libnvinfer.so.8`）插到 `LD_LIBRARY_PATH` 最前。apt 安装时该目录通常已是 `/usr/lib/x86_64-linux-gnu`（与 cuDNN 8 相同）。

### 5.5 Python：ORT 1.17.1 + `device=tensorrt`

```bash
# 建议独立 venv；若占用现场 1.18.1 环境，测完 TensorRT 后要装回 1.18.1 才能走 §4 CUDA EP
pip uninstall -y onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu==1.17.1 \
  -i https://mirrors.aliyun.com/pypi/simple/ \
  --trusted-host mirrors.aliyun.com

python3 - <<'PY'
import onnxruntime as ort, pathlib
print(ort.__version__)
print(ort.get_available_providers())
# 期望：1.17.1 且含 TensorrtExecutionProvider、CUDAExecutionProvider
root = pathlib.Path(ort.__file__).parent
hits = list(root.rglob("*tensorrt*.so")) + list(root.rglob("*Tensorrt*"))
print("trt libs:", hits[:8] or "(providers 可能打在 capi 内)")
PY
```

确认 TensorRT EP 链的是 **`.so.8`**：

```bash
python3 - <<'PY'
import onnxruntime as ort, pathlib, glob, os
root = pathlib.Path(ort.__file__).parent
cands = glob.glob(str(root / "**/*tensorrt*"), recursive=True)
print("\n".join(cands) or "no tensorrt so next to wheel")
PY
# 若找到 libonnxruntime_providers_tensorrt.so：
# ldd <该路径> | grep nvinfer
# 期望：libnvinfer.so.8 => ...   不能是 .so.10
```

`config/predict.json`：

```json
"device": "tensorrt",
"trt_fp16_enable": true,
"trt_engine_cache_path": "models/trt_cache"
```

V100 有 FP16 Tensor Core，保持 `trt_fp16_enable=true`。首次建 session 会 **编译 engine**（可能数分钟），缓存写入 `trt_engine_cache_path`，之后应明显变快。

建引擎时常见 **WARNING**（不是失败，不必重导 ONNX）：

```text
Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.
```

YOLO 图里的 gather/shape 常带 INT64，TRT 8.6 会降成 INT32。出现后请等 `session providers=['TensorrtExecutionProvider', ...]`；只有随后报 **unsupported operator / SM 70** 才需要另处理。

```bash
cd ~/scanly/predict_go
source ./env_cuda11.sh
python3 server_onnx.py
```

**成功标志：**

```text
[ort] version=1.17.1 available=[..., 'TensorrtExecutionProvider', 'CUDAExecutionProvider', ...]
[ort] TensorRT EP fp16=True cache=models/trt_cache
[ort] session providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

性能测试（`defects/test/test_onnx_infer_perf.py` 里 `DEVICE = "tensorrt"`）：

```bash
source ./env_cuda11.sh
python3 ../test/test_onnx_infer_perf.py
```

**不要**在 ORT 1.18.1 环境下测 `DEVICE=tensorrt`（会再次撞上 SM 70 或找不到 `.so.10`）。

TensorRT 性能请用 **FP32 密文**（`models/v1.3-onnx/`）+ `trt_fp16_enable=true`。不要拿 FP16 ONNX 再叠 TRT FP16：V100 上会出现 INT64 警告之后输出 NaN，解码 `round` 崩溃。`test_onnx_infer_perf.py` 在 `DEVICE=tensorrt` 时默认跳过 FP16 路。

和 CUDA EP 比：同一张图 FP32 CUDA 的 `session.run` 约 28ms，TensorRT 约 **6.6ms** 即属正常。`load_ms` 四五分钟是**第一次建引擎**，缓存进 `models/trt_cache` 后再次启动应到秒级。

### 5.6 ONNX 密文：默认不重导

`models/v1.3-onnx/*.onnx.enc`（及 fp16 目录）可直接喂给 TensorRT 8.6 parser。  
仅当日志出现 **unsupported operator / opset** 时，再考虑用更低 opset 重导；那是 parser 问题，不是 SM 70 问题。

### 5.7 Go 当前不能走 TensorRT 8.6

`scanly-defects-go-ort118` 绑定的是 **ORT C API 18**（ORT 1.18.1）。ORT **1.17.1 只提供 API 17**，该二进制加载不了。  
仓库也没有「ORT 1.17 + TensorRT 8.6」的第三套 flavor。

因此 V100 上：

| 进程 | 建议 |
|:---|:---|
| Python `server_onnx.py` / 性能脚本 | **§5**：ORT 1.17.1 + TensorRT 8.6 + `device=tensorrt` |
| Go `scanly-defects-go-ort118` | 保持 **§4 / §6**：ORT 1.18.1 + `device=cuda` |

不要把 1.17.1 的 `third_party/onnxruntime` 塞给 `*-ort118` 二进制。

---

## 6. Go：下载 ORT 1.18.1（CUDA 11）动态库


发布页：<https://github.com/microsoft/onnxruntime/releases/tag/v1.18.1>

| 包名 | 用途 |
|:---|:---|
| **`onnxruntime-linux-x64-gpu-1.18.1.tgz`** | **要这个**（CUDA **11**） |
| `onnxruntime-linux-x64-gpu-cuda12-1.18.1.tgz` | **不要**（仍要 cublasLt.so.12 / cuDNN 9 路线） |

```bash
cd /tmp
wget -O ort118.tgz \
  https://github.com/microsoft/onnxruntime/releases/download/v1.18.1/onnxruntime-linux-x64-gpu-1.18.1.tgz
tar -xzf ort118.tgz
cd ~/scanly/predict_go
rm -rf third_party/onnxruntime.bak
mv third_party/onnxruntime third_party/onnxruntime.bak 2>/dev/null || true
mkdir -p third_party/onnxruntime
cp -a /tmp/onnxruntime-linux-x64-gpu-1.18.1/lib/* third_party/onnxruntime/

cd third_party/onnxruntime
ln -sf libonnxruntime.so.1.18.1 libonnxruntime.so.1
ln -sf libonnxruntime.so.1.18.1 libonnxruntime.so
ls -l libonnxruntime.so* libonnxruntime_providers_cuda.so
```

启动 Go 前同样：

```bash
cd ~/scanly/predict_go
source ./env_cuda11.sh
export LD_LIBRARY_PATH="$(pwd)/third_party/onnxruntime:$LD_LIBRARY_PATH"
# 使用配对二进制（日志应有 flavor=ort118）
./dist/scanly-defects-go-ort118
# 或拷到 PATH 后直接：scanly-defects-go-ort118
```

本地编译双包：

```bash
cd scanly/defects/predict_go
bash tools/build_dual.sh
# dist/scanly-defects-go        → ORT ≥1.24
# dist/scanly-defects-go-ort118 → ORT 1.18.1（API 18 / onnxruntime_go v1.11.0）
```

若启动报 `requested API version [24] is not available`：说明仍在用默认 `scanly-defects-go`，**请改用 `scanly-defects-go-ort118`**。

---

## 7. 故障速查

| 现象 | 原因 | 处理 |
|:---|:---|:---|
| `libcublasLt.so.11: No such file` | 只有 CUDA 12 | 装 **cuda-toolkit-11-8**，`LD_LIBRARY_PATH` 含 `cuda-11.8/lib64` |
| `libcudnn.so.8` 找不到 / No such file | 未装 cuDNN 8 | `apt-get install libcudnn8`，确认 `/usr/lib/x86_64-linux-gnu/libcudnn.so.8` |
| `ls /usr/local/cudnn-8.9-cuda11` 不存在 | 用了 apt 却去找 tar 路径 | apt 路径见 **§2.1**；`source env_cuda11.sh`（默认已指向系统目录） |
| `libcudnn.so` → `.so.9` | 与 cuDNN 9 并存的正常现象 | **不要**强改全局软链；确认进程链的是 `libcudnn.so.8`（`ldd`） |
| session 只有 `CPUExecutionProvider` | CUDA EP 加载失败被降级 | 看上面几条；`server_onnx.py` 在 `device=cuda` 时会直接报错退出 |
| `CUDNN_FE` / `no kernel image` | 仍在用 ORT ≥1.20 或 CUDA12 包 | 确认 `ort.__version__==1.18.1`，且 Go 库不是 cuda12 / 1.23 包 |
| `Target GPU SM 70 is not supported` | V100 + TensorRT **10** | **§5**：卸 TRT 10，装 **8.6** + Python **ORT 1.17.1**；不要重导 ONNX |
| `libnvinfer.so.10: cannot open shared object file` | ORT 1.18.1 在找 TRT 10 | V100 不要补 10.x。CUDA EP 用 1.18.1；TensorRT 用 **§5**（1.17.1 + `.so.8`） |
| `libnvinfer.so.8` 找不到 | 未装 TensorRT 8.6 或仍被 10.x 占用 | **§5.2–5.3**；`ldconfig -p \| grep nvinfer` 应只有 `.so.8` |
| `apt-get install libnvinfer8` 装到 cuda12/cuda13 包 | CUDA 网络源版本更高 | 钉 `'libnvinfer8=*+cuda11.8'` 或直接装 local repo deb |
| `INT64 weights` / `cast down to INT32`（WARNING） | TRT 不原生支持 INT64 | **忽略**，等建完 engine；不要因此重导 ONNX |
| TensorRT 下 FP16 ONNX 解码 `cannot convert float NaN to integer` | FP16 ONNX 再开 `trt_fp16` | 改用 **FP32 密文 + TRT FP16**；不要叠两层半精度 |
| TensorRT 下 FP16 ONNX 解码 `NaN to integer` | FP16 ONNX 再开 `trt_fp16` | 改用 **FP32 密文 + TRT FP16**；不要叠两层半精度 |
| pip 很慢 | 默认 PyPI | 用阿里云镜像（§4.1） |
| 装了 11.8 仍找 `.so.12` | `LD_LIBRARY_PATH` 未含 11.8 或顺序不对 | `cuda-11.8/lib64` 写进 `LD_LIBRARY_PATH` 并重新开终端/`source` |
| `GET was unable to find an engine`（`convert_2_onnx.py` / Ultralytics export） | 推理用的 CUDA11 `LD_LIBRARY_PATH` 覆盖了 PyTorch wheel 自带 cuDNN | 见 **§3.3**：`LD_LIBRARY_PATH= python3 convert_2_onnx.py`，或脚本默认 sanitize；或 `HALF=False` |

---

## 8. 与 `run_go.md` 的分工

| 文档 | 内容 |
|:---|:---|
| [`run_go.md`](./run_go.md) | 编译、License、HTTP、软链、双二进制、通用 GPU 对齐、**转 ONNX 加密**、T4/A10 等卡的 **TensorRT 10** |
| **本文 `run_go_cuda11.md`** | V100 / FE 规避：**CUDA 11.8 + cuDNN 8 + ORT 1.18.1**（CUDA EP）；**§5 TensorRT 8.6 + ORT 1.17.1**；**§3.3** 与训练导出隔离 |

产线 CUDA EP：启动脚本统一 `source env_cuda11.sh`，`predict.json` 的 `device` 保持 `cuda`（ORT **1.18.1**）。  
产线 V100 TensorRT：Python 按 **§5**（ORT **1.17.1** + TensorRT 8.6，`device=tensorrt`）；Go 仍用 `device=cuda`。  
**训练导出 ONNX 不要**与推理共用同一套已 `source` 的 CUDA11 环境（§3.3）。
