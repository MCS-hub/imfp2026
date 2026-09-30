#!/usr/bin/env bash
#SBATCH --job-name=imfp_final_meadow
#SBATCH --account=project_462001620
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --chdir=/scratch/project_462001620/luuhau/imfp_image
#SBATCH --output=outs/reward_final_meadow-%j.log
#SBATCH --error=outs/reward_final_meadow-%j.log
#SBATCH --export=ALL
set -eo pipefail
srun --cpu-bind=none --nodes=1 --ntasks=1 \
  singularity exec --bind /scratch/project_462001620:/scratch/project_462001620 \
  /appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260124_092648/lumi-multitorch-full-u24r64f21m43t29-20260124_092648.sif bash -c '
    set -eo pipefail
    source /scratch/project_462001620/luuhau/py-meanflow-main/flow/bin/activate
    export PYTHONPATH="/scratch/project_462001620/luuhau/imfp_image/image_packages${PYTHONPATH:+:$PYTHONPATH}"
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
    export PYTHON="$(command -v python)"
    cd /scratch/project_462001620/luuhau/imfp_image
    exec bash scripts/run_reward_final.sh configs/reward_final_golden_retriever_meadow_tau200_r8_l8_v1.json
  '
