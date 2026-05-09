#!/usr/bin/env bash
# Export and INT4-quantize Qwen3-Coder-30B-A3B-Instruct to OpenVINO IR.
#
# Run this on a host with enough disk for ~70 GB of intermediate weights and
# enough RAM (32 GB+) to hold the source FP16 model during quantization.
# The NUC works but is slow; running this on a workstation and rsync'ing the
# output to the NUC is fine — the resulting IR is hardware-portable.
#
# Output: ${OUT_DIR} contains the .xml/.bin pair plus tokenizer files,
# ready to be served by OVMS or loaded by openvino_genai.LLMPipeline.

set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
OUT_DIR="${OUT_DIR:-./models/qwen3-coder-30b-a3b-int4}"
WEIGHT_FORMAT="${WEIGHT_FORMAT:-int4_sym}"
GROUP_SIZE="${GROUP_SIZE:-128}"
RATIO="${RATIO:-1.0}"
DATASET="${DATASET:-}"   # set to "wikitext2" to enable AWQ/data-aware quant

if ! command -v optimum-cli >/dev/null 2>&1; then
  echo "optimum-cli not found. Install with:"
  echo "  pip install --upgrade 'optimum[openvino]' nncf"
  exit 1
fi

mkdir -p "$(dirname "${OUT_DIR}")"

echo "Exporting ${MODEL_ID} to ${OUT_DIR}"
echo "  weight-format: ${WEIGHT_FORMAT}"
echo "  group-size:    ${GROUP_SIZE}"
echo "  ratio:         ${RATIO}"
[ -n "${DATASET}" ] && echo "  dataset:       ${DATASET} (data-aware)"

ARGS=(
  export openvino
  --model "${MODEL_ID}"
  --task text-generation-with-past
  --weight-format "${WEIGHT_FORMAT}"
  --group-size "${GROUP_SIZE}"
  --ratio "${RATIO}"
  --trust-remote-code
)

if [ -n "${DATASET}" ]; then
  ARGS+=(--dataset "${DATASET}" --awq --scale-estimation)
fi

ARGS+=("${OUT_DIR}")

optimum-cli "${ARGS[@]}"

echo
echo "Done. Output:"
du -sh "${OUT_DIR}"
ls -la "${OUT_DIR}"
echo
echo "Verify the model loads on the target device with:"
echo "  python3 ${PWD}/../bench/bench.py --model ${OUT_DIR} --device GPU"
