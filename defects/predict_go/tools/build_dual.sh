#!/usr/bin/env bash
# 发布双二进制：
#   scanly-defects-go           — onnxruntime_go v1.26（API 24，配 ORT ≥1.24）
#   scanly-defects-go-ort118    — onnxruntime_go v1.11.0（API 18，配 ORT 1.18.1；V100 + CUDA11/cuDNN8）
#
# 用法：
#   cd scanly/defects/predict_go && bash tools/build_dual.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT_DIR="${OUT_DIR:-$ROOT/dist}"
mkdir -p "$OUT_DIR"

EXT=""
if [[ "$(go env GOOS)" == "windows" ]]; then
  EXT=".exe"
fi

BAK_MOD="${OUT_DIR}/.go.mod.build.bak"
BAK_SUM="${OUT_DIR}/.go.sum.build.bak"

restore_mod() {
  if [[ -f "$BAK_MOD" ]]; then
    mv -f "$BAK_MOD" go.mod
  fi
  if [[ -f "$BAK_SUM" ]]; then
    mv -f "$BAK_SUM" go.sum
  fi
}
trap restore_mod EXIT

build_one() {
  local flavor="$1"
  local modfile="$2"
  local sumfile="$3"
  local outname="$4"

  echo "==> building flavor=${flavor} out=${outname}"
  cp go.mod "$BAK_MOD"
  cp go.sum "$BAK_SUM"
  if [[ "$modfile" != "go.mod" ]]; then
    cp "$modfile" go.mod
    cp "$sumfile" go.sum
  fi

  CGO_ENABLED=1 go build \
    -ldflags "-X main.ortFlavor=${flavor}" \
    -o "${OUT_DIR}/${outname}${EXT}" \
    ./cmd/server

  restore_mod
  echo "    ok: ${OUT_DIR}/${outname}${EXT}"
}

build_one "ort-new" "go.mod" "go.sum" "scanly-defects-go"
build_one "ort118" "go.mod.ort118" "go.sum.ort118" "scanly-defects-go-ort118"

trap - EXIT

echo
echo "done. binaries in ${OUT_DIR}:"
ls -la "${OUT_DIR}/scanly-defects-go"*"${EXT}" 2>/dev/null || true
echo
echo "现场选用（V100 / CUDA11）："
echo "  1) source env_cuda11.sh"
echo "  2) third_party 放 ORT 1.18.1 GPU（非 cuda12 包）"
echo "  3) 运行 scanly-defects-go-ort118${EXT} ，predict.json device=cuda"
echo "  新卡/ORT≥1.24：scanly-defects-go${EXT}"
