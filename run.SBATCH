#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=47:59:59
#SBATCH --mem=10GB
#SBATCH --gres=gpu
#SBATCH --job-name=torch

module purge

singularity exec --nv \
	    --overlay /scratch/hl3797/GACV-F2021/pytorch1.8.0-cuda11.1.ext3:ro \
	    /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python train.py $1 > $1.log 2>&1"