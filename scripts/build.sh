#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
NN_CXX_COMPILER="${NN_CXX_COMPILER:-/usr/bin/g++}"
BUILD_JOBS="${BUILD_JOBS:-$(command -v nproc >/dev/null 2>&1 && nproc || echo 4)}"
CMAKE_GENERATOR="${CMAKE_GENERATOR:-Unix Makefiles}"

if [[ "${1:-}" == "--clean" ]]; then
  rm -rf "${BUILD_DIR}"
fi

if [[ ! -f "${BUILD_DIR}/Makefile" ]]; then
  mkdir -p "${BUILD_DIR}"
  CXX="${NN_CXX_COMPILER}" cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
    -G "${CMAKE_GENERATOR}" \
    -DNN_FORCE_MODERN_GCC=ON
fi

make -C "${BUILD_DIR}" -j"${BUILD_JOBS}"

echo "Build complete: ${BUILD_DIR}"
