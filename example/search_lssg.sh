#!/usr/bin/env bash
set -euo pipefail

# Edit this section for your dataset and experiment.
LSSG_USE_MINHASH=ON       # Must match the variant used to build the index.
LSSG_ENABLE_NATIVE_SIMD=ON
BASE_VECS="/path/to/ytb_audio/ytb_audio_base.fvecs"
BASE_LABELS="/path/to/ytb_audio/label_base.txt"
QUERY_VECS="/path/to/ytb_audio/ytb_audio_query_containment.fvecs"
QUERY_LABELS="/path/to/ytb_audio/ytb_audio_query_containment.txt"
SPACE="l2"                # l2 or ip
INDEX_FILE="./index/ytb_audio_16_128_l2_label3862_minhash.poi"
SCOPE_FILE="./index/ytb_audio_label3862_minhash.sco"
FILTER="containment"      # containment, equality, or overlap
GROUND_TRUTH="./gt/ytb_audio_k10/containment.bin"
K=10
EFS="50,100,200,500,1000,2000,3000,5000"
MAX_BASE_VECTORS=0         # Must match MAX_VECTORS in build_lssg.sh.
MAX_QUERIES=0              # 0 uses the complete query set

if (( $# != 0 )); then
  echo "This script takes no command-line arguments; edit the settings at the top of the file." >&2
  exit 2
fi

case "$FILTER" in
  containment) GT_OPTION=--gt_containment ;;
  equality) GT_OPTION=--gt_equality ;;
  overlap) GT_OPTION=--gt_overlap ;;
  *) echo "Unsupported FILTER: $FILTER" >&2; exit 2 ;;
esac

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
BUILD_DIR="$REPO_DIR/build"

cmake -S "$REPO_DIR" -B "$BUILD_DIR" \
  -DLSSG_USE_MINHASH="$LSSG_USE_MINHASH" \
  -DLSSG_ENABLE_NATIVE_SIMD="$LSSG_ENABLE_NATIVE_SIMD" >&2
cmake --build "$BUILD_DIR" --target search_lssg -j >&2

if [[ ! -f "$GROUND_TRUTH" ]]; then
  echo "Ground truth not found; generating $GROUND_TRUTH" >&2
  cmake --build "$BUILD_DIR" --target gen_label_gt -j >&2
  "$BUILD_DIR/bin/gen_label_gt" \
    --basevec "$BASE_VECS" \
    --queryvec "$QUERY_VECS" \
    --base_labelset "$BASE_LABELS" \
    --query_labelset "$QUERY_LABELS" \
    --gt_file "$GROUND_TRUTH" \
    --space "$SPACE" \
    --type "$FILTER" \
    --k "$K" \
    --max_base_vectors "$MAX_BASE_VECTORS" \
    --max_queries "$MAX_QUERIES" >&2
fi

ARGS=(
  --query_vec "$QUERY_VECS"
  --query_labelset "$QUERY_LABELS"
  --index_location "$INDEX_FILE"
  --scope_location "$SCOPE_FILE"
  --space "$SPACE"
  --k "$K"
  --base_vec "$BASE_VECS"
  --base_labelset "$BASE_LABELS"
  --max_base_vectors "$MAX_BASE_VECTORS"
  "$GT_OPTION" "$GROUND_TRUTH"
  --efs "$EFS"
)
if [[ "$MAX_QUERIES" != 0 ]]; then
  ARGS+=(--max_queries "$MAX_QUERIES")
fi

"$BUILD_DIR/bin/search_lssg" "${ARGS[@]}"
