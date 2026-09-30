#!/usr/bin/env bash
# CUDA 11.8 + cuDNN 8 运行时路径（与 CUDA 12 / cuDNN 9 并排）。
# 用法：source env_cuda11.sh
# 详细步骤：scanly/doc/99-ops/run_go_cuda11.md
#
# 默认按 apt 安装 libcudnn8（库在 /usr/lib/x86_64-linux-gnu）。
# 若改用 tar 解压，启动前：
#   export CUDNN8_LIB=/usr/local/cudnn-8.9-cuda11/lib
#
# TensorRT 8.6（可选，V100 TensorRT EP，见 run_go_cuda11.md §5）：
# apt 后库与 cuDNN8 同目录；tar 时：
#   export TENSORRT86_LIB=/usr/local/TensorRT-8.6.1/lib

CUDA11_HOME="${CUDA11_HOME:-/usr/local/cuda-11.8}"
CUDNN8_LIB="${CUDNN8_LIB:-/usr/lib/x86_64-linux-gnu}"
TENSORRT86_LIB="${TENSORRT86_LIB:-}"

if [[ ! -d "$CUDA11_HOME/lib64" ]]; then
  echo "警告: 未找到 $CUDA11_HOME/lib64 ，请先安装 cuda-toolkit-11-8" >&2
fi

if [[ ! -e "$CUDNN8_LIB/libcudnn.so.8" ]]; then
  # tar 包常见 lib / lib64
  if [[ -e /usr/local/cudnn-8.9-cuda11/lib/libcudnn.so.8 ]]; then
    CUDNN8_LIB=/usr/local/cudnn-8.9-cuda11/lib
  elif [[ -e /usr/local/cudnn-8.9-cuda11/lib64/libcudnn.so.8 ]]; then
    CUDNN8_LIB=/usr/local/cudnn-8.9-cuda11/lib64
  else
    echo "警告: 未找到 libcudnn.so.8 。请 apt-get install libcudnn8 或设置 CUDNN8_LIB" >&2
  fi
fi

export CUDA_HOME="$CUDA11_HOME"
export PATH="$CUDA11_HOME/bin${PATH:+:$PATH}"
export LD_LIBRARY_PATH="$CUDNN8_LIB:$CUDA11_HOME/lib64:$CUDA11_HOME/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ -z "$TENSORRT86_LIB" ]]; then
  if [[ -e /usr/lib/x86_64-linux-gnu/libnvinfer.so.8 ]]; then
    TENSORRT86_LIB=/usr/lib/x86_64-linux-gnu
  elif [[ -e /usr/local/TensorRT-8.6.1/lib/libnvinfer.so.8 ]]; then
    TENSORRT86_LIB=/usr/local/TensorRT-8.6.1/lib
  fi
fi
if [[ -n "$TENSORRT86_LIB" && -e "$TENSORRT86_LIB/libnvinfer.so.8" ]]; then
  if [[ "$TENSORRT86_LIB" != "$CUDNN8_LIB" ]]; then
    export LD_LIBRARY_PATH="$TENSORRT86_LIB:$LD_LIBRARY_PATH"
  fi
  echo "TENSORRT86_LIB=$TENSORRT86_LIB"
  ls -l "$TENSORRT86_LIB"/libnvinfer.so.8 "$TENSORRT86_LIB"/libnvinfer.so.8.* 2>/dev/null | head -3
elif [[ -n "${TENSORRT86_LIB}" ]]; then
  echo "警告: TENSORRT86_LIB=$TENSORRT86_LIB 但未找到 libnvinfer.so.8" >&2
fi
if ldconfig -p 2>/dev/null | grep -q 'libnvinfer.so.10'; then
  echo "警告: 系统仍有 libnvinfer.so.10。V100 TensorRT EP 须用 8.6（.so.8），请按 run_go_cuda11.md §5 卸掉 TensorRT 10。" >&2
fi

echo "CUDA_HOME=$CUDA_HOME"
echo "CUDNN8_LIB=$CUDNN8_LIB"
echo "LD_LIBRARY_PATH 前缀: $CUDNN8_LIB : $CUDA11_HOME/lib64"
ls "$CUDA11_HOME/lib64"/libcublasLt.so.11 2>/dev/null || echo "缺少 libcublasLt.so.11"
ls -l "$CUDNN8_LIB"/libcudnn.so.8 "$CUDNN8_LIB"/libcudnn.so.8.* 2>/dev/null | head -5 || echo "缺少 libcudnn.so.8*"
# 提示：libcudnn.so 指向 .so.9 是与 cuDNN9 并存时的正常现象，不必改
if [[ -L "$CUDNN8_LIB/libcudnn.so" ]]; then
  echo "提示: libcudnn.so -> $(readlink -f "$CUDNN8_LIB/libcudnn.so" 2>/dev/null || readlink "$CUDNN8_LIB/libcudnn.so") （ORT 1.18 应使用 .so.8）"
fi
