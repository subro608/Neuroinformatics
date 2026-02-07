#!/bin/bash
#SBATCH --job-name=eeg_test
#SBATCH --output=/scratch/%u/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/%u/slurm_logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=torch_pr_60_tandon_advanced
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1

set -euo pipefail

CONTAINERS_DIR="/scratch/${USER}/containers"
IMAGE="${CONTAINERS_DIR}/w2v2_cuda_cu128.sif"
REPO_ROOT="/scratch/${USER}/Neuroinformatics"
TEST_SCRIPT="${REPO_ROOT}/testing/test_multiscale_graph_spectral_advanced.py"
DATA_DIR="${REPO_ROOT}/datasets/model_data"
MODEL_PATH="${EEG_MODEL_PATH:-${REPO_ROOT}/spatial_spectral_20260206_201256/models/best_model_overall.pth}"

LOGDIR="/scratch/${USER}/slurm_logs"
mkdir -p "${LOGDIR}"

export PYTHONPATH="${REPO_ROOT}/datasets:${REPO_ROOT}/training:${REPO_ROOT}/layers:${REPO_ROOT}/utils:${REPO_ROOT}/data_provider:${REPO_ROOT}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "===== TEST CONFIG ====="
echo "Node:    $(hostname)"
echo "Model:   ${MODEL_PATH}"
echo "Data:    ${DATA_DIR}"
echo "DIM:     ${EEG_DIM:-64}"
nvidia-smi || true
echo "======================="

singularity exec --nv \
    --home "/scratch/${USER}" \
    --bind "/scratch/${USER}:/scratch/${USER}" \
    --env "USER=${USER},PYTHONPATH=${PYTHONPATH},PYTHONUNBUFFERED=1,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,EEG_DATA_DIR=${DATA_DIR},EEG_MODEL_PATH=${MODEL_PATH},EEG_DIM=${EEG_DIM:-64}" \
    "${IMAGE}" \
    python3 "${TEST_SCRIPT}"

echo "Done. Test complete."
