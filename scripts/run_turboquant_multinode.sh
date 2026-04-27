#!/usr/bin/bash
# Launch turboquant_ep llama4 demo across all nodes in a mint job.
# Designed to be invoked via `mint exec` so this script runs on every node
# simultaneously; torchrun's c10d rendezvous assigns node_rank automatically.
#
# Required env:
#   MASTER_HOST   hostname of the rendezvous master (any one node will do)
# Optional env:
#   NNODES        default: 2
#   NPROC         default: 4   (4 GB200 per node)
#   EP            default: 8   (cross-node EP -- the point of turboquant)
#   STEPS         default: 50
#   LOCAL_BS      default: (config-defined; override e.g. LOCAL_BS=4)
#   SEQ_LEN       default: (config-defined; override e.g. SEQ_LEN=4096)
#   CONFIG        default: turboquant_llama4_debugmodel
#                 Other options:
#                   baseline_llama4_debugmodel        (apples-to-apples baseline, debug)
#                   turboquant_llama4_17bx16e         (Scout 17Bx16E with TQ fwd+bwd)
#                   turboquant_fwd_only_llama4_17bx16e (Scout, fwd-only TQ ablation)
#                   baseline_llama4_17bx16e           (Scout baseline, no TQ)
#   RUN_TAG       default: tq_demo. Used as rdzv_id; bump it (e.g. RUN_TAG=baseline)
#                 when launching back-to-back runs in the same session so the
#                 c10d rendezvous server doesn't reuse stale state.
#   TURBOQUANT_DIR  default: <repo>/third_party/turboquant (resolved relative to this script)
#   CONDA_DIR     default: /packages/torchtitan_conda_gb200/conda

set -ex

: "${MASTER_HOST:?set MASTER_HOST to the rendezvous node hostname}"
NNODES=${NNODES:-2}
NPROC=${NPROC:-4}
EP=${EP:-8}
STEPS=${STEPS:-50}
CONFIG=${CONFIG:-turboquant_llama4_debugmodel}
RUN_TAG=${RUN_TAG:-tq_demo}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
TURBOQUANT_DIR=${TURBOQUANT_DIR:-${REPO_DIR}/third_party/turboquant}
CONDA_DIR=${CONDA_DIR:-/packages/torchtitan_conda_gb200/conda}

# Run from repo root so `python -m torchtitan.train` picks up the synced repo,
# not the older torchtitan bundled in the conda env's site-packages.
cd "${REPO_DIR}"

# --- MAST CUDA / fbcode platform setup ---------------------------------------
# Replicates what MAST job runners normally export so the conda env can find
# libcuda / libnvidia-ml / libnvshmem from /usr/local/fbcode. GB200 is aarch64
# so we always take the aarch64 branch here.
PLATFORM="platform010"
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    PLATFORM="platform010-aarch64"
fi
LIBCUDA="/usr/local/fbcode/${PLATFORM}/lib/libcuda.so"
export LIBCUDA_DIR="${LIBCUDA%/*}"
export TRITON_LIBCUDA_PATH="/usr/local/fbcode/${PLATFORM}/lib/"
PRELOAD_PATH="${PRELOAD_PATH:=$LIBCUDA:/usr/local/fbcode/${PLATFORM}/lib/libnvidia-ml.so:/usr/local/fbcode/${PLATFORM}/lib/libnvidia-ptxjitcompiler.so}"
if [ "$ARCH" = "aarch64" ]; then
    GCC_DIR=$(ls -d "${CONDA_DIR}/libexec/gcc/aarch64-conda-linux-gnu/"*/ 2>/dev/null | head -1)
    if [ -n "$GCC_DIR" ]; then
        export PATH="${GCC_DIR}:${PATH}"
    fi
    mkdir -p /tmp/cuda_stubs
    ln -sf "/usr/local/fbcode/${PLATFORM}/lib/libcuda.so.1" /tmp/cuda_stubs/libcuda.so.1
    export LD_LIBRARY_PATH="/tmp/cuda_stubs:${LD_LIBRARY_PATH:-}"
    PRELOAD_PATH="${PRELOAD_PATH}:/usr/local/fbcode/${PLATFORM}/lib/cuda-no-rpath-13.0/libnvtx3interop.so.1"
fi
export LD_PRELOAD="${PRELOAD_PATH}"
# Resolve the python-version-specific nvshmem lib dir (env is python3.12 here,
# but glob keeps this robust if conda gets rebuilt on a different python).
NVSHMEM_LIB=$(ls -d ${CONDA_DIR}/lib/python*/site-packages/nvidia/nvshmem/lib 2>/dev/null | head -1)
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${CONDA_DIR}/lib${NVSHMEM_LIB:+:${NVSHMEM_LIB}}"
# -----------------------------------------------------------------------------

# Put the repo first so its torchtitan wins over the conda-env-installed one.
export PYTHONPATH=${REPO_DIR}:${TURBOQUANT_DIR}:${PYTHONPATH:-}
export PYTORCH_ALLOC_CONF="expandable_segments:True"

EXTRA_ARGS=()
[ -n "${LOCAL_BS:-}" ] && EXTRA_ARGS+=("--training.local-batch-size=${LOCAL_BS}")
[ -n "${SEQ_LEN:-}" ]  && EXTRA_ARGS+=("--training.seq-len=${SEQ_LEN}")
[ -n "${SEED:-}" ]     && EXTRA_ARGS+=("--debug.seed=${SEED}")

torchrun \
  --nnodes=${NNODES} \
  --nproc_per_node=${NPROC} \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_HOST}:29501 \
  --rdzv_id=${RUN_TAG} \
  --local-ranks-filter 0 --role rank --tee 3 \
  -m torchtitan.train \
  --module turboquant_ep \
  --config ${CONFIG} \
  --parallelism.expert_parallel_degree=${EP} \
  --training.steps=${STEPS} \
  "${EXTRA_ARGS[@]}"
