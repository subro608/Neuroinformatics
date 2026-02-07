#!/bin/bash
#SBATCH --job-name=eeg_mvt
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
# MVT EEG Classification - Multi-GPU Training (DataParallel)
# ============================================================

CONTAINERS_DIR="/scratch/${USER}/containers"
IMAGE="${CONTAINERS_DIR}/w2v2_cuda_cu128.sif"
REPO_ROOT="/scratch/${USER}/Neuroinformatics"
TRAIN_SCRIPT="${REPO_ROOT}/training/train_kfold_mvt.py"
DATA_DIR="${REPO_ROOT}/datasets/model_data"

LOGDIR="/scratch/${USER}/slurm_logs"
mkdir -p "${LOGDIR}"

# Stage data to local SSD
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    LOCAL_DATA="${SLURM_TMPDIR}/model_data"
    echo "===== LOCAL STAGING ====="
    echo "Copying data to node-local SSD..."
    rsync -a "${DATA_DIR}/" "${LOCAL_DATA}/"
    echo "Staging complete. Size: $(du -sh "${LOCAL_DATA}" | cut -f1)"
    echo "========================="
    DATA_DIR="${LOCAL_DATA}"
fi

# Environment variables
export PYTHONPATH="${REPO_ROOT}/datasets:${REPO_ROOT}/training:${REPO_ROOT}/layers:${REPO_ROOT}/utils:${REPO_ROOT}/data_provider:${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

export HF_HOME="/scratch/${USER}/.cache/huggingface"
mkdir -p "${HF_HOME}"

export WANDB_PROJECT="${WANDB_PROJECT:-eeg_spatial_spectral}"
export WANDB_DIR="${WANDB_DIR:-/scratch/${USER}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/scratch/${USER}/wandb_cache}"
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}"

# Training overrides
export EEG_DATA_DIR="${DATA_DIR}"
export EEG_NUM_GPUS="${EEG_NUM_GPUS:-4}"
export EEG_BATCH_SIZE="${EEG_BATCH_SIZE:-16}"
export EEG_EPOCHS="${EEG_EPOCHS:-200}"
export EEG_LR="${EEG_LR:-0.0003}"
export EEG_DIM="${EEG_DIM:-512}"
export EEG_NUM_FOLDS="${EEG_NUM_FOLDS:-5}"

# Safety checks
echo "===== ENVIRONMENT CHECK ====="
echo "Node:    $(hostname)"
echo "JobID:   ${SLURM_JOB_ID:-local}"
echo "Image:   ${IMAGE}"
echo "Data:    ${DATA_DIR}"
echo "Script:  ${TRAIN_SCRIPT}"
nvidia-smi || true
echo "============================="

if [[ ! -f "${IMAGE}" ]]; then
    echo "ERROR: Container image not found: ${IMAGE}"
    exit 1
fi
if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "ERROR: Training script not found: ${TRAIN_SCRIPT}"
    exit 1
fi
if [[ ! -f "${DATA_DIR}/labels.json" ]]; then
    echo "ERROR: Data not found at ${DATA_DIR}/labels.json"
    echo "Run datasets/data_prep.py first!"
    exit 1
fi

echo "===== TRAINING CONFIG ====="
echo "Model:      MVT EEG"
echo "GPUs:       ${EEG_NUM_GPUS}"
echo "Batch size: ${EEG_BATCH_SIZE} (per GPU)"
echo "Epochs:     ${EEG_EPOCHS}"
echo "LR:         ${EEG_LR}"
echo "Model dim:  ${EEG_DIM}"
echo "CV folds:   ${EEG_NUM_FOLDS} (subject-aware GroupKFold)"
echo "Data dir:   ${DATA_DIR}"
echo "==========================="

# Bind mounts
BIND_ARGS=(
    --bind "/scratch/${USER}:/scratch/${USER}"
)
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    BIND_ARGS+=(--bind "${SLURM_TMPDIR}:${SLURM_TMPDIR}")
fi

# Launch training
set -x
singularity exec --nv \
    --home "/scratch/${USER}" \
    "${BIND_ARGS[@]}" \
    --env "USER=${USER},PYTHONPATH=${PYTHONPATH},PYTHONUNBUFFERED=${PYTHONUNBUFFERED},WANDB_PROJECT=${WANDB_PROJECT},WANDB_DIR=${WANDB_DIR},WANDB_CACHE_DIR=${WANDB_CACHE_DIR},OMP_NUM_THREADS=${OMP_NUM_THREADS},MKL_NUM_THREADS=${MKL_NUM_THREADS},EEG_DATA_DIR=${EEG_DATA_DIR},EEG_NUM_GPUS=${EEG_NUM_GPUS},EEG_BATCH_SIZE=${EEG_BATCH_SIZE},EEG_EPOCHS=${EEG_EPOCHS},EEG_LR=${EEG_LR},EEG_DIM=${EEG_DIM},EEG_NUM_FOLDS=${EEG_NUM_FOLDS}" \
    "${IMAGE}" \
    python3 "${TRAIN_SCRIPT}"
set +x

echo "Done. MVT training complete."
echo "Check WandB project: ${WANDB_PROJECT}"
echo "SLURM logs: ${LOGDIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.{out,err}"
