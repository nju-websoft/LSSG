#!/usr/bin/env bash
set -euo pipefail

# Edit this section for your dataset and experiment.
LSSG_USE_MINHASH=ON       # ON: LSSG-MinHash, OFF: LSSG-IVF
LSSG_ENABLE_NATIVE_SIMD=ON
BASE_VECS="/path/to/ytb_audio/ytb_audio_base.fvecs"
BASE_LABELS="/path/to/ytb_audio/label_base.txt"
SPACE="l2"                # l2 or ip
INDEX_FILE="./index/ytb_audio_16_128_l2_label3862_minhash.poi" # graph definition
SCOPE_FILE="./index/ytb_audio_label3862_minhash.sco" # label set definition
M=16
EFC=128
THREADS=16
MAX_VECTORS=0              # 0 uses the complete base set

if (( $# != 0 )); then
  echo "This script takes no command-line arguments; edit the settings at the top of the file." >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
BUILD_DIR="$REPO_DIR/build"

cmake -S "$REPO_DIR" -B "$BUILD_DIR" \
  -DLSSG_USE_MINHASH="$LSSG_USE_MINHASH" \
  -DLSSG_ENABLE_NATIVE_SIMD="$LSSG_ENABLE_NATIVE_SIMD" >&2
cmake --build "$BUILD_DIR" --target build_lssg -j >&2

ARGS=(
  --basevec "$BASE_VECS"
  --labelset "$BASE_LABELS"
  --space "$SPACE"
  --m "$M"
  --efc "$EFC"
  --threads "$THREADS"
  --index_location "$INDEX_FILE"
  --scope_location "$SCOPE_FILE"
)
if [[ "$MAX_VECTORS" != 0 ]]; then
  ARGS+=(--max_vectors "$MAX_VECTORS")
fi

"$BUILD_DIR/bin/build_lssg" "${ARGS[@]}"
