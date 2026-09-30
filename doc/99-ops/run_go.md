# Go 加密 ONNX 推理服务运行说明

实现入口：`scanly/defects/predict_go/cmd/server`。帮助文档：`scanly/defects/predict_go/README.md`。接口契约与 Python 版相同：`scanly/doc/02-dr/缺陷推理/01.HTTP接口文档.md`。

**V100 / CUDNN_FE 规避（CUDA 11 + ORT 1.18.1）：** 见同目录 [`run_go_cuda11.md`](./run_go_cuda11.md)。

## 依赖

- Go 1.22+（`go.mod` 声明 `1.24.0`，推荐安装 1.24.x）
- CGO + C 编译器（依赖 `github.com/yalue/onnxruntime_go`，**必须** `CGO_ENABLED=1`）
- [ONNX Runtime](https://github.com/microsoft/onnxruntime/releases) 动态库（运行时加载；编译期不链入，但现场必须提供）
- 厂商侧 Python：`pip install -r scanly/train/requirements.txt` 与 `scanly/defects/predict_go/tools/requirements.txt`

把平台对应的 ORT 库放到默认目录，或在 `config/predict.json` 填写 `ort_lib_path`：

- Linux / macOS：`third_party/onnxruntime/`（`libonnxruntime.so` / `.dylib`）
- Windows：`third_party_win/onnxruntime/`（`onnxruntime.dll`）

## Go 环境准备（编译）

本节只覆盖 **Ubuntu** 与 **Windows** 上为编译本服务准备的环境。编译产物可拷到目标机；目标机仍需对应平台的 ONNX Runtime 动态库。

### 通用检查

```bash
go version          # 期望 go1.22 及以上，推荐 1.24.x
go env CGO_ENABLED  # 必须为 1
```

若为 `0`，先装好下面的 C 编译器，再执行：

```bash
# Linux / macOS / Git Bash
export CGO_ENABLED=1

# Windows CMD
set CGO_ENABLED=1

# Windows PowerShell
$env:CGO_ENABLED = "1"
```

拉取依赖并编译：

```bash
cd scanly/defects/predict_go
go mod download
go build -o scanly-defects-go ./cmd/server
```

Windows 建议输出带后缀：

```bat
go build -o scanly-defects-go.exe ./cmd/server
```

常见失败：`build constraints exclude all Go files in .../onnxruntime_go` → CGO 未启用或找不到 `gcc`。

---

### Ubuntu（编译环境）

#### 1. 安装 Go

推荐用官方包（版本可控），勿依赖过旧的 `apt` 默认包：

```bash
# 以 1.24.x 为例；版本号按 https://go.dev/dl/ 更新
cd /tmp
curl -fsSL -o go.tgz https://go.dev/dl/go1.24.5.linux-amd64.tar.gz
sudo rm -rf /usr/local/go
sudo tar -C /usr/local -xzf go1.24.5.linux-amd64.tar.gz
echo 'export PATH=/usr/local/go/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
go version
```

若已用包管理器安装且 `go version` 已满足要求，可跳过。

#### 1.1 模块下载超时（国内镜像）

若 `go mod download` / `go build` 访问 `proxy.golang.org` 超时（`i/o timeout`），设置国内代理后重试：

```bash
# 当前 shell 临时生效
export GOPROXY=https://goproxy.cn,direct
# 可选：校验和数据库也走国内（少数环境需要）
export GOSUMDB=sum.golang.google.cn

go env -w GOPROXY=https://goproxy.cn,direct
go env -w GOSUMDB=sum.golang.google.cn

cd ~/scanly/predict_go   # 或仓库内 scanly/defects/predict_go
go mod download
go build -o scanly-defects-go ./cmd/server
```

备选镜像：`https://goproxy.io,direct`，或阿里云 `https://mirrors.aliyun.com/goproxy/,direct`。  
`go env -w` 会写入用户配置，之后新开终端也生效。用 `go env GOPROXY` 确认。

#### 2. 安装 C 工具链（CGO）

```bash
sudo apt update
sudo apt install -y build-essential
gcc --version
```

确认：

```bash
go env CGO_ENABLED   # 应为 1；若为 0：export CGO_ENABLED=1
which gcc
```

#### 3. （可选）交叉编译到 Windows

本机 Ubuntu 打 Windows 二进制时，除 Go 外还需 MinGW：

```bash
sudo apt install -y gcc-mingw-w64
cd scanly/defects/predict_go
CGO_ENABLED=1 CC=x86_64-w64-mingw32-gcc GOOS=windows GOARCH=amd64 \
  go build -o scanly-defects-go.exe ./cmd/server
```

产物在 Windows 上运行，仍需 Windows 版 `onnxruntime.dll`（见下节 ORT）。

#### 4. ONNX Runtime（Linux，编译机可一并准备）

从 [Releases](https://github.com/microsoft/onnxruntime/releases) 下载对应架构包。解压后 **实体库** 为 `libonnxruntime.so.1.x.x`；`libonnxruntime.so` / `libonnxruntime.so.1` 多为软链接，上传服务器后常丢失，需在目标机重建：

```bash
cd third_party/onnxruntime
ln -sf libonnxruntime.so.1.28.1 libonnxruntime.so.1
ln -sf libonnxruntime.so.1.28.1 libonnxruntime.so
```

或配置 `ort_lib_path` 指向实体文件。**完整步骤见下文「Ubuntu 运行」§2。**

GPU 推理还需现场已安装匹配的 NVIDIA 驱动 / CUDA（与所选 ORT GPU 包说明一致）。

---

### Windows（编译环境）

#### 1. 安装 Go

1. 打开 [https://go.dev/dl/](https://go.dev/dl/)，下载 **Windows / amd64** 的 MSI（推荐 1.24.x）。
2. 安装时勾选加入 PATH。
3. 新开 **CMD** 或 **PowerShell**：

```bat
go version
```

#### 2. 安装 C 编译器（CGO，必需）

Windows 上 Go 的 CGO 依赖 **MinGW-w64 的 gcc**（MSVC 不能直接给 CGO 用）。任选一种：

**方式 A：MSYS2（推荐）**

1. 安装 [MSYS2](https://www.msys2.org/)。
2. 在 MSYS2 UCRT64 终端执行：

```bash
pacman -Syu
pacman -S --needed mingw-w64-ucrt-x86_64-gcc
```

3. 将 `C:\msys64\ucrt64\bin`（按实际安装路径）加入系统 **PATH**。
4. 新开 CMD / PowerShell：

```bat
gcc --version
go env CGO_ENABLED
```

应为 `1`；若仍为 `0`：

```bat
set CGO_ENABLED=1
```

**方式 B：Chocolatey**

```bat
choco install mingw -y
```

安装后确认 `gcc` 在 PATH 中，再 `set CGO_ENABLED=1` 后编译。

#### 3. 编译

在仓库内（CMD 示例）：

```bat
cd scanly\defects\predict_go
set CGO_ENABLED=1
go mod download
go build -o scanly-defects-go.exe .\cmd\server
```

PowerShell：

```powershell
cd scanly\defects\predict_go
$env:CGO_ENABLED = "1"
go mod download
go build -o scanly-defects-go.exe .\cmd\server
```

#### 4. ONNX Runtime（Windows）

从 [Releases](https://github.com/microsoft/onnxruntime/releases) 下载：

- CPU：`onnxruntime-win-x64-*.zip`
- GPU：官方 Win-x64 GPU 包

将 `onnxruntime.dll`（GPU 包还需同目录下的 `execution_providers_*.dll` 等依赖）放到 **`third_party_win\onnxruntime\`**（与 Linux 的 `third_party` 隔离），或在 `config\predict.json` 填写 `ort_lib_path`。

运行时可将该目录加入 PATH，或保证进程能按配置路径加载 DLL。

---

## Ubuntu 运行（现场 / 本机）

本节面向 **已有 Linux 二进制** 的启动，**不要求** 安装 Go / gcc。若本机既编译又运行，可先按上文「Ubuntu（编译环境）」打出 `scanly-defects-go`，再按本节准备运行文件。

### 1. 目录与交付物

建议保持服务根目录结构（可用环境变量 `SCANLY_PREDICT_GO_ROOT` 指向该目录）：

```text
predict_go/                          # 或任意安装目录
├── scanly-defects-go                # Linux 二进制
├── config/
│   ├── predict.json
│   └── license.json
├── models/
│   ├── v1.3-onnx/                 # 全精度 FP32
│   │   ├── xxx.onnx.enc
│   │   └── xxx.meta.json
│   └── v1.3-onnx-fp16/            # 半精度 FP16（可选）
│       ├── xxx.onnx.enc
│       └── xxx.meta.json
├── third_party/onnxruntime/
│   ├── libonnxruntime.so.1.28.1     # 实体库（版本号以实际为准）
│   ├── libonnxruntime.so -> …       # 软链接（服务默认加载名）
│   └── libonnxruntime.so.1 -> …     # 软链接（可选，对齐官方包）
└── （可选）与 Python 共用的 scheme：../predict/config/scheme.json
```

最少需要：二进制、`predict.json`、`license.json`、密文+meta、ORT 动态库、方案 `scheme.json`。

### 2. 安装 ONNX Runtime（运行时必需）

1. 从 [ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases) 下载：
   - CPU：`onnxruntime-linux-x64-*.tgz`
   - GPU：官方 Linux GPU 包（`device` 用 `cuda` 时）
   - **选包前先读 §2.1**：CUDA 主版本、GPU `compute_cap`（如 V100S=7.0）必须与包匹配；V100 建议 **ORT 1.18.1**，勿盲目最新/1.20。
2. 解压后 `lib/` 目录常见文件：

| 文件 | 说明 |
|:---|:---|
| `libonnxruntime.so.1.x.x` | **实体库**（必须拷到服务器） |
| `libonnxruntime.so` | **软链接** → 通常指向 `.so.1` |
| `libonnxruntime.so.1` | **软链接** → 通常指向 `.so.1.x.x` |
| `libonnxruntime_providers_*.so` | GPU 包额外依赖（`cuda` 时一并拷贝） |

用 scp/FTP 上传时软链接常会丢失，服务器上往往只剩 `libonnxruntime.so.1.x.x`。服务默认查找名为 **`libonnxruntime.so`** 的文件，需在服务器上重建链接（或改配置）。

3. 放到 `third_party/onnxruntime/` 后重建软链接（版本号按实际文件改）：

```bash
cd /path/to/predict_go/third_party/onnxruntime
# 确认实体库已在此目录，例如 libonnxruntime.so.1.18.1（V100）或其它版本号
ln -sf libonnxruntime.so.1.18.1 libonnxruntime.so.1
ln -sf libonnxruntime.so.1.18.1 libonnxruntime.so
ls -l
# 期望类似：
# libonnxruntime.so -> libonnxruntime.so.1.18.1
# libonnxruntime.so.1 -> libonnxruntime.so.1.18.1
# libonnxruntime.so.1.18.1
```

4. 也可不建链接，在 `config/predict.json` 写实体路径：

```json
"ort_lib_path": "third_party/onnxruntime/libonnxruntime.so.1.18.1"
```

若加载失败，可把该目录加入动态库搜索路径后再启动：

```bash
export LD_LIBRARY_PATH="$(pwd)/third_party/onnxruntime:${LD_LIBRARY_PATH:-}"
```

GPU 包还需本机 NVIDIA 驱动（及 ORT 说明对应的 CUDA 运行时）；先确认：

```bash
nvidia-smi
```

`device` 为 `cuda` 时，把同版本 GPU 包中的 `libonnxruntime_providers_cuda.so`、`libonnxruntime_providers_shared.so` 等一并放到同一目录。

**cuDNN（CUDA 推理必需）**：仅有 CUDA Toolkit 不够，ORT CUDA EP 还要 `libcudnn.so`。你本机是 **CUDA 12.8**，装 **cuDNN 9 for CUDA 12**。

#### 方式 A：apt 发行版包（推荐）

机器上若已能 `apt` 装 CUDA（已有 NVIDIA 源），直接：

```bash
sudo apt-get update
# 搜一下本机源里实际包名
apt-cache search cudnn | head -40

# 官方元包（CUDA 12）
sudo apt-get -y install cudnn9-cuda-12
# 若上面找不到，可只装运行库：
# sudo apt-get -y install libcudnn9-cuda-12
```

若提示 `Unable to locate package`，说明还没加 NVIDIA 网络源。按 [CUDA Ubuntu 安装指南](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) 配置好 `cuda-keyring` / NVIDIA repo 后再执行上面命令。官方说明：[cuDNN Linux 安装](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/linux.html)。

装完后：

```bash
ldconfig -p | grep -i cudnn
# 常见：libcudnn.so.9 -> ...
# ORT 若找 libcudnn.so（无版本后缀），补软链（路径以 ldconfig 输出为准）：
sudo ln -sf /usr/lib/x86_64-linux-gnu/libcudnn.so.9 /usr/lib/x86_64-linux-gnu/libcudnn.so
sudo ldconfig

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
```

再重启 `scanly-defects-go` 测预测。

#### 方式 B：官网 tar 包（无 apt 源时）

1. 打开 [cuDNN 下载页](https://developer.nvidia.com/cudnn)（需 NVIDIA 账号），选 **Linux x86_64、CUDA 12.x、tar**。
2. 解压后把 `lib` 拷到 CUDA 目录，例如：

```bash
tar -xvf cudnn-linux-x86_64-*-cuda12-*.tar.xz
sudo cp -P cudnn-*/lib/libcudnn* /usr/local/cuda/lib64/
sudo ldconfig
```

未装 cuDNN 时可暂设 `"device": "cpu"`，服务可预测但不用 GPU。

### 2.1 GPU / ORT / CUDA / cuDNN 版本注意事项（必读）

`device=cuda` 时，**驱动、CUDA、cuBLAS、cuDNN、ORT GPU 包必须同一代对齐**。Go 二进制本身不内嵌 CUDA；现场换卡或换包都要重新核对下表。

#### 先查本机

```bash
nvidia-smi
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
nvcc --version                    # Toolkit 主版本，如 12.8
ldconfig -p | grep -i cublasLt    # 如 libcublasLt.so.12
ldconfig -p | grep -i cudnn       # 如 libcudnn.so.9
```

| 输出 | 含义 |
|:---|:---|
| `name` | 如 `Tesla V100S-PCIE-32GB` |
| `compute_cap` | GPU 算力，如 V100 = **7.0**（Volta / sm_70） |
| `cublasLt.so.N` | N 须与 ORT GPU 包要求的 CUDA 主版本一致（见下） |

#### 对齐原则

| 组件 | 要求 |
|:---|:---|
| `predict.json` → `device` | 写 **`cuda` 或 `gpu`**（二者均启用 CUDA EP；写 `cpu` 才走 CPU） |
| **Go 绑定 ↔ ORT 动态库** | **必须匹配**（见下表）。`go.mod` 里的 `onnxruntime_go` 会请求固定的 ORT C API 版本；动态库太旧会启动时报 API version 不匹配 |
| ORT 包 | 选 **GPU** 包，且说明面向的 CUDA 主版本与本机一致（本机 CUDA 12 → 选 CUDA **12** 的 ORT，不要选要 `libcublasLt.so.13` 的 CUDA 13 包） |
| 目录 | Linux：`third_party/onnxruntime/`；Windows：`third_party_win/onnxruntime/`；**主库与 providers 同版本、勿混拷** |
| cuBLAS | 来自 CUDA Toolkit；`libcublasLt.so.12` 配 CUDA 12 |
| cuDNN | CUDA 12 用 **cuDNN 9 for CUDA 12**；ORT 还要能解析到 `libcudnn.so`（常需 `.so.9` → `.so` 软链） |
| GPU 算力 | ORT/cuDNN 预编译内核须覆盖本机 `compute_cap`；否则预测阶段报 `no kernel image` |

#### Go 绑定与 ORT 版本（当前工程）

本仓库 `go.mod`：`github.com/yalue/onnxruntime_go v1.26.0`，初始化时会请求 **ORT C API version 24**。

| 现场 `libonnxruntime` | 能否直接用当前已编译的 `scanly-defects-go` |
|:---|:---|
| ORT **1.24.x / 1.25.x / 1.26+**（提供 API ≥24） | 可以（再单独核对 CUDA / 算力） |
| ORT **1.20.x / 1.19.x** 配默认 `scanly-defects-go` | **不可以**（API 不够）。典型报错见下 |
| 更旧的 ORT | 不可以 |

典型报错（你遇到的正是这类）：

```text
The requested API version [24] is not available, only API versions [1, 18] are supported in this build. Current ORT Version is: 1.18.1
ONNX Runtime 初始化失败: ... Error setting ORT API base
```

含义：**不是**「Go 语言不兼容老卡」，而是 **二进制与 ORT 动态库 API 未配对**。

可选路径（**方案 B：发布两个二进制**）：

| 现场 ORT 动态库 | 使用的可执行文件 | Go 绑定 |
|:---|:---|:---|
| **ORT ≥ 1.24**（API 24） | `scanly-defects-go` | `onnxruntime_go v1.26`（`go.mod`） |
| **ORT 1.18.1**（API 18，**V100 优先**） | `scanly-defects-go-ort118` | `onnxruntime_go v1.11.0`（`go.mod.ort118`） |

本地一次打出两个包：

```bash
cd scanly/defects/predict_go
bash tools/build_dual.sh
# 产物在 dist/：scanly-defects-go 、 scanly-defects-go-ort118
```

启动日志会打印 `flavor=ort-new` 或 `flavor=ort118`，便于核对。**二进制与 third_party 中的 ORT 主版本必须配对**，不要混用。

（不推荐）方案 A：只发一个二进制并降绑定——已由方案 B 替代。

#### 产线常见卡：Tesla V100S（compute_cap 7.0，必须 GPU）

1. V100 为 **Volta sm_70**。1.19/1.20/1.23 在本机（驱动 580 + CUDA 12.8 + cuDNN 9）上已验证易挂 Conv；**优先 ORT 1.18.1 GPU（CUDA 11 + cuDNN 8）**，详见 [`run_go_cuda11.md`](./run_go_cuda11.md)。
2. **推荐路径**：
   - `pip install onnxruntime-gpu==1.18.1` + `source env_cuda11.sh`，Python 冒烟通
   - 官方包 **`onnxruntime-linux-x64-gpu-1.18.1.tgz`**（**非** `*-cuda12-*`）→ `third_party/onnxruntime/`
   - 二进制 **`scanly-defects-go-ort118`**（`flavor=ort118`）
   - `predict.json`：`"device": "cuda"`（或 `"gpu"`）。CUDA EP 失败时**不再静默回退 CPU**。
3. 新卡 / ORT≥1.24：用 **`scanly-defects-go`** + 对应 GPU 库。
4. 换包须整包替换，勿混旧版 `providers_*.so`。

#### 错误 → 处理速查

| 日志关键词 | 原因 | 处理 |
|:---|:---|:---|
| `requested API version [24] is not available` | 默认二进制配了 1.18 库 | **`scanly-defects-go-ort118` + ORT 1.18.1 GPU** |
| `libcublasLt.so.11` … No such file | 缺 CUDA 11 用户态库 | 见 [`run_go_cuda11.md`](./run_go_cuda11.md) |
| `CUDNN_FE` / `CUDNN failure 5003` | 仍在用 ≥1.20 / CUDA12+cuDNN9 | 切回 **1.18.1 + CUDA11 + cuDNN8** |
| 无日志 `CUDA EP 已启用` | `device` 仍为 `cpu` | 改为 `cuda`/`gpu` |

```bash
# flavor=ort118 ；third_party 为 1.18.1 GPU（CUDA11 包）全套 so
ln -sf libonnxruntime.so.1.18.1 libonnxruntime.so.1
ln -sf libonnxruntime.so.1.18.1 libonnxruntime.so
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/lib64:$(pwd)/third_party/onnxruntime:${LD_LIBRARY_PATH:-}
source ./env_cuda11.sh
```

启动前统一用 CUDA11 环境（勿再用仅 CUDA12 的路径）：

```bash
source ./env_cuda11.sh
export LD_LIBRARY_PATH="$(pwd)/third_party/onnxruntime:$LD_LIBRARY_PATH"
```

### 3. 配置 `predict.json`

编辑 `config/predict.json`（路径相对服务根目录）：

| 字段 | 说明 |
|:---|:---|
| `host` / `port` | 默认 `0.0.0.0` / `37871` |
| `infer_type` | `obb` 或 `detect` |
| `obb_model_enc_path` / `obb_meta_path` | 密文与 meta（detect 用对应 `model_enc_path` / `detect_meta_path`） |
| `license_path` | 默认 `config/license.json` |
| `scheme_path` | 默认 `../predict/config/scheme.json` |
| `ort_lib_path` | 空=按 OS 自动：Linux `third_party/onnxruntime/libonnxruntime.so`；Windows `third_party_win/onnxruntime/onnxruntime.dll` |
| `device` | `cpu` / `cuda`/`gpu` / `tensorrt`（别名 `trt`、`trt_fp16`）。`tensorrt` 启用 TensorRT EP（默认 FP16）+ CUDA 作算子回退；要求会话实际走 TensorRT |
| `trt_fp16_enable` | 默认 `true`；仅 `device=tensorrt` 时生效 |
| `trt_engine_cache_path` | 默认 `models/trt_cache`；首次建引擎较慢，后续复用 |

**TensorRT 注意**：`available` 里有 `TensorrtExecutionProvider` 不等于能加载成功。ORT 还需系统侧 `libnvinfer.so.*`（ORT 1.18.1 GPU 包常见要 **`.so.10`**）。若日志出现 `libnvinfer.so.10: cannot open shared object file`，ORT 会静默回退 CUDA；本服务在 `device=tensorrt` 时会判定失败。

**V100 / SM 70 不能用 TensorRT 10**：Tesla V100 是 Volta `compute_cap 7.0`。TensorRT 10.x 已去掉 SM 70（官方矩阵从 **SM 7.5 / T4** 起）。日志 `Target GPU SM 70 is not supported by this TensorRT release` 出现时，**不要重导 ONNX**。CUDA EP 请保持 `"device": "cuda"`。若现场必须 TensorRT，见 [`run_go_cuda11.md`](./run_go_cuda11.md) **§5**（**TensorRT 8.6** + Python **ORT 1.17.1**，现有密文不用重导；与 ORT 1.18.1 找 `.so.10` 不是同一条线）。

### 安装 TensorRT 10（补齐 `libnvinfer.so.10`）

先确认驱动 / CUDA。TensorRT 运行库必须与当前推理线的 CUDA 主版本对齐（可与并排 Toolkit 共存）：

| 推理线 | 应下载的 local repo | 禁止装到的包 |
|:---|:---|:---|
| ORT **1.18.1** + CUDA **11.8**（且 GPU ≥ SM 7.5，如 T4/A10） | TensorRT 10.x **cuda-11.8**（如 `...-10.9.0-cuda-11.8`） | `*+cuda12*` / `*+cuda13*` |
| ORT ≥1.20 / CUDA **12**（Ampere/Ada/Hopper） | TensorRT 10.x **cuda-12.x** | `*+cuda13*`、以及 CUDA 11 的 TRT |
| **V100 / SM 70** | 不要装 TensorRT 10；见 [`run_go_cuda11.md`](./run_go_cuda11.md) **§5** | 任何 TensorRT 10.x |

```bash
nvidia-smi
ldconfig -p | grep nvinfer || true
```

**方式 A（推荐，系统库）：NVIDIA deb 本地源**

1. 打开 [TensorRT Download](https://developer.nvidia.com/tensorrt)（需登录），选 **TensorRT 10.x**、对应 Ubuntu（如 22.04）与上表 CUDA 版本的 *local repo deb*。  
2. 安装 local repo 并导入 keyring（包名/路径以下载文件为准）：

```bash
sudo dpkg -i nv-tensorrt-local-repo-ubuntu2204-10.9.0-cuda-11.8_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-ubuntu2204-10.9.0-cuda-11.8/nv-tensorrt-local-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
```

3. **必须钉版本**。机器上若已配置 NVIDIA CUDA 网络源（`developer.download.nvidia.com` / `.nvidia.cn`），其中常带更新的 `libnvinfer10`（例如 `10.16.1.11-1+cuda13.2`）。不钉版本时 `apt-get install libnvinfer10` 会优先装网络源高版本，体积约 1.8GB 且与 CUDA 11.8 / ORT 1.18.1 不匹配。

先看候选，确认 local repo 的 `10.9.0.*+cuda11.8` 在列：

```bash
apt-cache policy libnvinfer10
apt-cache madison libnvinfer10
```

再按 CUDA 后缀安装（ORT 1.18.1 线用 `cuda11.8`；CUDA 12 线改成 `cuda12.x` 实际后缀）：

```bash
sudo apt-get install -y \
  'libnvinfer10=*+cuda11.8' \
  'libnvinfer-plugin10=*+cuda11.8' \
  'libnvinfer-vc-plugin10=*+cuda11.8' \
  'libnvonnxparsers10=*+cuda11.8'
sudo apt-mark hold \
  libnvinfer10 libnvinfer-plugin10 libnvinfer-vc-plugin10 libnvonnxparsers10
```

若通配仍被网络源抢走，直接装 local repo 目录里的 deb（完全绕过 CUDA 源）：

```bash
REPO=/var/nv-tensorrt-local-repo-ubuntu2204-10.9.0-cuda-11.8
sudo apt-get install -y \
  "$REPO"/libnvinfer10_*+cuda11.8_amd64.deb \
  "$REPO"/libnvinfer-plugin10_*+cuda11.8_amd64.deb \
  "$REPO"/libnvinfer-vc-plugin10_*+cuda11.8_amd64.deb \
  "$REPO"/libnvonnxparsers10_*+cuda11.8_amd64.deb
sudo apt-mark hold \
  libnvinfer10 libnvinfer-plugin10 libnvinfer-vc-plugin10 libnvonnxparsers10
```

4. 验证（版本串须含 `10.9.0` 与 `cuda11.8`，**不能**是 `10.16` / `cuda13.2`）：

```bash
dpkg -l | grep -E 'libnvinfer|libnvonnxparsers'
ldconfig -p | grep nvinfer
# 应能看到 libnvinfer.so.10
```

**方式 B（免 root / 多版本）：tar 包**

从同一下载页取 Linux tar（CUDA 12），解压后：

```bash
tar -xzf TensorRT-10.*.Linux.x86_64-gnu.cuda-12.*.tar.gz
export LD_LIBRARY_PATH=/path/to/TensorRT-10.x.x.x/lib:${LD_LIBRARY_PATH:-}
# 可写入 ~/.bashrc 或现场启动脚本
ls /path/to/TensorRT-10.x.x.x/lib/libnvinfer.so.10
```

**方式 C（仅 Python 侧快速试）：pip**

```bash
python3 -m pip install -U 'tensorrt-cu12'
python3 - <<'PY'
import glob, os, tensorrt as trt
print("tensorrt", trt.__version__)
roots = list(dict.fromkeys([os.path.dirname(trt.__file__), *(__import__("site").getsitepackages())]))
for r in roots:
    hits = glob.glob(r + "/**/libnvinfer.so*", recursive=True)
    if hits:
        print("libs:", hits[:5])
        print("export LD_LIBRARY_PATH=%s:$LD_LIBRARY_PATH" % os.path.dirname(hits[0]))
        break
PY
# 按上面打印的目录 export 后再跑 perf
```

装好后跑性能测试（不要 `source env_cuda11.sh` 盖掉 CUDA12/TRT 路径，除非你已确认两套库共存无冲突）：

```bash
export LD_LIBRARY_PATH=...   # 若用 tar/pip，确保含 libnvinfer.so.10
cd /root/scanly
# test_onnx_infer_perf.py 中 DEVICE = "tensorrt"
python3 test/test_onnx_infer_perf.py
```

期望：`session providers=['TensorrtExecutionProvider', ...]`。仍失败则把 `ldconfig -p | grep nvinfer` 与完整 ORT 日志留存排查。

未装 TensorRT 时请保持 `"device": "cuda"`。

### 4. 启动

```bash
cd /path/to/predict_go          # 或 export SCANLY_PREDICT_GO_ROOT=/path/to/predict_go
chmod +x ./scanly-defects-go
./scanly-defects-go
```

日志中应看到监听地址；License 通过且 ORT/模型加载成功时，方可预测。

### 5. 自检

```bash
curl -s http://127.0.0.1:37871/
curl -s http://127.0.0.1:37871/health/ready
curl -s http://127.0.0.1:37871/v1/license
```

本机路径预测图例（Python，不上传）：改 `scanly/defects/test/test_predict_path_api.py` 中 `BASE_URL` / `IMAGE_PATHS`（路径须为**服务端**可读绝对路径）后运行：

```bash
python scanly/defects/test/test_predict_path_api.py
```

### Python 冒烟（隔离验证 GPU，非产线）

`scanly/defects/predict_go/server_onnx.py` 读**同一套** `config/predict.json`、License、密文模型；推理用 **pip 的 `onnxruntime-gpu`**（不加载 `third_party` 下载库）。默认端口 **37872**。

```bash
cd ~/scanly/predict_go   # 或 defects/predict_go
pip uninstall -y onnxruntime
pip install onnxruntime-gpu cryptography pillow numpy
# predict.json 中 device=cuda
python server_onnx.py
# 另开终端：把 test_predict_path_api.py 的 BASE_URL 改成 http://<host>:37872 再测
```

- Python 通、Go 不通 → 查 Go 的下载包 / 绑定配对  
- Python 也挂（CUDNN_FE / no kernel image）→ 环境/卡问题，换 ORT 版本也救不了 Go

| 现象 | 常见原因 |
|:---|:---|
| ORT 初始化失败 | 缺少 `libonnxruntime.so`（上传后软链接丢失未重建），或 `ort_lib_path` / `LD_LIBRARY_PATH` 不对 |
| `/health/ready` 为 503 | License 未通过，或模型未加载 |
| License 无效 / GPU 不一致 | `license.json` 与本机 `nvidia-smi` 三列不符；无 GPU 开发机可启动但预测会授权失败 |
| `device=cuda` 相关失败（cublas / cudnn / no kernel image） | 见上文 **§2.1 GPU / ORT / CUDA / cuDNN 版本注意事项** |

### 6. 本机编译并运行（开发）

若在同一台 Ubuntu 上从源码跑：

```bash
cd scanly/defects/predict_go
# 先按「Ubuntu（编译环境）」装好 Go + build-essential，并放好 ORT
# 一次打出两个发布二进制（推荐）
bash tools/build_dual.sh
# 仅调试新 ORT 绑定时：
# go build -o scanly-defects-go ./cmd/server

# V100 + ORT 1.18.1：
./dist/scanly-defects-go-ort118
# 新 ORT（≥1.24）：
# ./dist/scanly-defects-go
```

---

## 配置

| 文件 | 作用 |
|:---|:---|
| `predict_go/config/predict.json` | 端口（默认 37871）、密文路径、License、ORT 库、`device`（`cpu`/`cuda`）、切片 |
| `predict_go/config/license.json` | 现场 License（Python `tools/gen_license.py` 生成） |
| `predict_go/models/*.onnx.enc` | 加密权重（`scanly/train/convert_2_onnx.py` 产出） |
| `../predict/config/scheme.json` | 默认方案（与 Python 共用） |

环境变量 `SCANLY_PREDICT_GO_ROOT` 可指定服务根目录。

## 启动（简表）

完整步骤见上文 **「Ubuntu 运行」**。Windows 现场将二进制与 `onnxruntime.dll` 放到对应目录后，运行 `scanly-defects-go.exe` 即可（配置字段相同）。

| 地址 | 用途 |
|:---|:---|
| `http://<host>:37871/` | 服务说明 JSON（不是 Gradio） |
| `http://<host>:37871/v1/predict` | 边端推理 |
| `http://<host>:37871/v1/license` | License / GPU 状态 |
| `http://<host>:37871/health/ready` | 就绪 |

无 NVIDIA GPU 的开发机可以启动并查方案，但 License 不会通过，预测返回授权失败。

## 转 ONNX 并加密（Ubuntu）

在训练机（Ubuntu）上，把 `.pt` 导出为加密 ONNX，供 Go 服务使用。

### 1. 依赖

```bash
python3 -m pip install -r scanly/train/requirements.txt
```

### 2. 改参数后运行

编辑 `scanly/train/convert_2_onnx.py` 底部变量，例如：

| 变量 | 含义 |
|:---|:---|
| `MODEL_PATH` | 训练得到的 `.pt`（相对 `scanly/train/` 或绝对路径） |
| `TASK` | `obb` 或 `detect` |
| `MODEL_ID` | 交付用模型名（影响输出文件名） |
| `HALF` | 默认 `True`：导出 FP16 ONNX（须 GPU）；`False` 为 FP32 |
| `DEVICE` | 导出设备；`HALF=True` 时用 `0` / `cuda`，不能 `cpu` |
| `SANITIZE_LD_LIBRARY_PATH` | 默认 `True`：导出前清空本进程 `LD_LIBRARY_PATH`（避免与 Go ORT 的 CUDA11 路径冲突） |
| `OUTPUT_DIR` | 默认 `scanly/train/output/onnx` |
| `AES_KEY_PATH` | 空=新建密钥；已有 `.key.json` 则复用（换机/续签时保持一致） |
| `KEEP_PLAIN_ONNX` | 默认 `True`（保留明文 `.onnx`，人工删除后再交付）；`False` 则自动删明文。原始 `.pt` **从不删除** |

```bash
python scanly/train/convert_2_onnx.py

# 若仍报 GET was unable to find an engine：
LD_LIBRARY_PATH= python3 scanly/train/convert_2_onnx.py
# 或 HALF=False 导出 FP32
```

半精度产物的 `meta.json` 含 `precision=fp16`；Go / Python 推理按 ONNX 实际 dtype 喂数，与既有 FP32 密文兼容。
### 3. 产物

输出目录（默认 `scanly/train/output/onnx/`）会得到：

| 文件 | 去向 |
|:---|:---|
| `{MODEL_ID}.onnx.enc` | 拷到 `predict_go/models/`，给现场 |
| `{MODEL_ID}.meta.json` | 同上 |
| `{MODEL_ID}.key.json` | **仅厂商留存**，签发 License 用；禁止随安装包分发 |

同步改 `predict_go/config/predict.json` 里的 `obb_model_enc_path` / `obb_meta_path`（或 detect 对应字段）指向上述密文与 meta。

---

## License 制作（本机）

在厂商本机签发即可（有/无 GPU 均可：GPU 信息可从现场抄写后填入）。依赖：

```bash
pip install -r scanly/defects/predict_go/tools/requirements.txt
```

脚本：`scanly/defects/predict_go/tools/gen_license.py`（改底部变量后直接运行）。

### 1. 首次生成签发密钥（仅一次）

设 `INIT_KEYS=True`，运行一次：

```bash
python scanly/defects/predict_go/tools/gen_license.py
```

会生成：

- 私钥 `predict_go/keys/license_ed25519.pem`（**勿分发**）
- 公钥写入 `predict_go/internal/license/license_ed25519.pub`（Go embed）

换过密钥后必须 **重新编译** Go 服务。之后把 `INIT_KEYS` 改回 `False`。

### 2. 登记现场 GPU

现场（或本机有卡时）执行：

```bash
nvidia-smi --query-gpu=uuid,name,serial --format=csv,noheader
```

或本机设 `LIST_LOCAL_GPUS = True` 跑脚本打印三列，再改回 `False`。

把输出填进脚本里的 `GPUS`（可多张卡）。授权条件：本机 **任一** GPU 的 uuid、name、serial **三项全匹配** 列表中某一条。

### 3. 签发

确认：

| 变量 | 含义 |
|:---|:---|
| `AES_KEY_PATH` | 上一步 `convert_2_onnx.py` 产出的 `.key.json` |
| `GPUS` | 现场 uuid / name / serial |
| `MODEL_ID` | 与导出一致（可空，则用 key 文件内 model_id） |
| `EXPIRES_AT` | 空=不过期；例 `2027-12-31T23:59:59+08:00` |
| `OUTPUT_PATH` | 默认 `predict_go/config/license.json` |

```bash
python scanly/defects/predict_go/tools/gen_license.py
```

得到 `config/license.json`（内含已签名的 AES 密钥），随现场安装包分发。

本机无卡调试：可填真实现场 GPU 信息签发；服务能启动，但本机预测仍会因 GPU 不一致失败（符合预期）。

---

## 厂商交付清单

发到现场：

- `{id}.onnx.enc`、`{id}.meta.json`、`license.json`
- Go 二进制（与 ORT **配对**）：`scanly-defects-go`（ORT≥1.24）或 `scanly-defects-go-ort118`（ORT **1.18.1**）
- 同代 ORT 动态库、`predict.json` / `scheme.json`

**不要**发送：`.pt`、明文 `.onnx`、`.key.json`、`keys/license_ed25519.pem`。
**不要**混用：例如 `scanly-defects-go` + ORT 1.18.1，或 `*-ort118` + ORT 1.20/1.24。
