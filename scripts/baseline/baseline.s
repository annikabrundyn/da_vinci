#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --time=08:10:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=20

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "

source /ext3/env.sh
conda activate da_vinci

python ../../src/models/baseline/baseline_model.py --data_dir /scratch/js11133/da_vinci/raw_data/
"
