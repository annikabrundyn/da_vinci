#!/bin/bash
#SBATCH --job-name=u0_u3
#SBATCH --output=array_%A_%a.out
#SBATCH --error=array_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --export=ALL
#SBATCH --time=72:00:00
#SBATCH --mem=20G
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --array=1-12

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "
source /ext3/env.sh
conda activate da_vinci

pip install fire
python ./test_sweep.py $SLURM_ARRAY_TASK_ID
"
