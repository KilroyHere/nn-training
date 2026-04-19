#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MNIST_DIR="${ROOT_DIR}/data/mnist"
BASE_URL="https://storage.googleapis.com/cvdf-datasets/mnist"

mkdir -p "${MNIST_DIR}"

download_and_extract() {
  local file="$1"
  local gz_file="${MNIST_DIR}/${file}.gz"
  local raw_file="${MNIST_DIR}/${file}"

  if [[ -f "${raw_file}" ]]; then
    return
  fi

  if [[ ! -f "${gz_file}" ]]; then
    curl -L --fail --retry 3 --retry-delay 1 \
      -o "${gz_file}" "${BASE_URL}/${file}.gz"
  fi

  gzip -dk "${gz_file}"
}

download_and_extract "train-images-idx3-ubyte"
download_and_extract "train-labels-idx1-ubyte"
download_and_extract "t10k-images-idx3-ubyte"
download_and_extract "t10k-labels-idx1-ubyte"

echo "MNIST files ready in ${MNIST_DIR}"
