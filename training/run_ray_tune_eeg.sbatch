#!/bin/bash
#SBATCH --job-name=eeg_ray_tune
#SBATCH --output=/scratch/%u/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/%u/slurm_logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --account=torch_pr_60_tandon_advanced
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:4

set -euo pipefail

# ============================================================
# Ray Tune HP Search for EEG Classification (MVT + Multiscale GS)
# ============================================================
# 4 GPUs = 4 parallel trials (1 GPU each)
# Uses ASHA scheduler + Optuna (TPE Bayesian optimization)
# ============================================================

# ----------------------------
# Container setup
# ----------------------------
CONTAINERS_DIR="/scratch/${USER}/containers"
# Try Ray container first, fall back to base container
if [[ -f "${CONTAINERS_DIR}/w2v2_cuda_cu128_ray.sif" ]]; then
    IMAGE="${CONTAINERS_DIR}/w2v2_cuda_cu128_ray.sif"
elif [[ -f "${CONTAINERS_DIR}/w2v2_cuda_cu128.sif" ]]; then
    IMAGE="${CONTAINERS_DIR}/w2v2_cuda_cu128.sif"
else
    echo "ERROR: No container found in ${CONTAINERS_DIR}"
    exit 1
fi

# ----------------------------
# Paths
# ----------------------------
REPO_ROOT="/scratch/${USER}/Neuroinformatics"
RAY_SCRIPT="${REPO_ROOT}/training/ray_tune_eeg.py"
DATA_DIR="${REPO_ROOT}/datasets/model_data"

LOGDIR="/scratch/${USER}/slurm_logs"
RAY_RESULTS_DIR="/scratch/${USER}/ray_results/eeg_$(date +%Y-%m-%d)"

mkdir -p "${LOGDIR}" "${RAY_RESULTS_DIR}"

# ----------------------------
# Stage data to local SSD
# ----------------------------
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    LOCAL_DATA="${SLURM_TMPDIR}/model_data"
    echo "===== LOCAL STAGING ====="
    echo "Copying data to node-local SSD..."
    rsync -a "${DATA_DIR}/" "${LOCAL_DATA}/"
    echo "Staging complete. Size: $(du -sh "${LOCAL_DATA}" | cut -f1)"
    echo "========================="
    DATA_DIR="${LOCAL_DATA}"
fi

# ----------------------------
# Environment variables
# ----------------------------
export PYTHONPATH="${REPO_ROOT}/datasets:${REPO_ROOT}/training:${REPO_ROOT}/layers:${REPO_ROOT}/utils:${REPO_ROOT}/data_provider:${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1

# Prevent BLAS/OMP oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# HuggingFace cache
export HF_HOME="/scratch/${USER}/.cache/huggingface"
mkdir -p "${HF_HOME}"

# WandB configuration
export WANDB_PROJECT="${WANDB_PROJECT:-eeg_ray_tune}"
export WANDB_DIR="${WANDB_DIR:-/scratch/${USER}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/scratch/${USER}/wandb_cache}"
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}"

# Ray temp directory (use job-specific dir to avoid stale session conflicts)
export RAY_TMPDIR="/scratch/${USER}/ray_tmp/${SLURM_JOB_ID:-local}"
mkdir -p "${RAY_TMPDIR}"
export RAY_RESULTS_DIR="${RAY_RESULTS_DIR}"

# ----------------------------
# Ray Tune controls (overridable via env vars)
# ----------------------------
TUNE_NUM_SAMPLES="${TUNE_NUM_SAMPLES:-50}"
TUNE_MAX_EPOCHS="${TUNE_MAX_EPOCHS:-100}"
TUNE_GPUS_PER_TRIAL="${TUNE_GPUS_PER_TRIAL:-1}"
TUNE_CPUS_PER_TRIAL="${TUNE_CPUS_PER_TRIAL:-4}"
TUNE_FOLD="${TUNE_FOLD:-0}"

# ----------------------------
# Safety checks
# ----------------------------
echo "===== ENVIRONMENT CHECK ====="
echo "Node:    $(hostname)"
echo "JobID:   ${SLURM_JOB_ID:-local}"
echo "Image:   ${IMAGE}"
echo "Data:    ${DATA_DIR}"
echo "Script:  ${RAY_SCRIPT}"
nvidia-smi || true
echo "============================="

if [[ ! -f "${RAY_SCRIPT}" ]]; then
    echo "ERROR: Ray Tune script not found: ${RAY_SCRIPT}"
    exit 1
fi
if [[ ! -f "${DATA_DIR}/labels.json" ]]; then
    echo "ERROR: Data not found at ${DATA_DIR}/labels.json"
    echo "Run datasets/data_prep.py first!"
    exit 1
fi

echo "===== RAY TUNE CONFIG ====="
echo "GPUs allocated:     4 (L40S)"
echo "Parallel trials:    $(( 4 / TUNE_GPUS_PER_TRIAL ))"
echo "Total trials:       ${TUNE_NUM_SAMPLES}"
echo "Max epochs/trial:   ${TUNE_MAX_EPOCHS}"
echo "GPUs/trial:         ${TUNE_GPUS_PER_TRIAL}"
echo "CPUs/trial:         ${TUNE_CPUS_PER_TRIAL}"
echo "Fold index:         ${TUNE_FOLD}"
echo "Data dir:           ${DATA_DIR}"
echo "Results dir:        ${RAY_RESULTS_DIR}"
echo "WandB project:      ${WANDB_PROJECT}"
echo "==========================="

# ----------------------------
# Bind mounts
# ----------------------------
BIND_ARGS=(
    --bind "/scratch/${USER}:/scratch/${USER}"
)
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    BIND_ARGS+=(--bind "${SLURM_TMPDIR}:${SLURM_TMPDIR}")
fi

# ----------------------------
# Launch Ray Tune inside Singularity container
# ----------------------------
set -x
singularity exec --nv \
    --home "/scratch/${USER}" \
    "${BIND_ARGS[@]}" \
    --env "USER=${USER},PYTHONPATH=${PYTHONPATH},PYTHONUNBUFFERED=${PYTHONUNBUFFERED},WANDB_PROJECT=${WANDB_PROJECT},WANDB_DIR=${WANDB_DIR},WANDB_CACHE_DIR=${WANDB_CACHE_DIR},OMP_NUM_THREADS=${OMP_NUM_THREADS},MKL_NUM_THREADS=${MKL_NUM_THREADS},EEG_DATA_DIR=${DATA_DIR},RAY_RESULTS_DIR=${RAY_RESULTS_DIR},RAY_TMPDIR=${RAY_TMPDIR}" \
    "${IMAGE}" \
    python3 "${RAY_SCRIPT}" \
        --num_samples "${TUNE_NUM_SAMPLES}" \
        --max_epochs "${TUNE_MAX_EPOCHS}" \
        --gpus_per_trial "${TUNE_GPUS_PER_TRIAL}" \
        --cpus_per_trial "${TUNE_CPUS_PER_TRIAL}" \
        --data_dir "${DATA_DIR}" \
        --fold "${TUNE_FOLD}"
set +x

echo "Done. Ray Tune EEG HP search complete."
echo "Results dir: ${RAY_RESULTS_DIR}"
echo "WandB project: ${WANDB_PROJECT}"
echo "SLURM logs: ${LOGDIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.{out,err}"
