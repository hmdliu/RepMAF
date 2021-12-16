## RepMAF: When Re-parameterization Meets Multi-scale Attention

CSCI-GA 2271: Computer Vision - Final Project

## Setup
**Requisites:** numpy, torch, torchvision, torchsummary.
Clone the project repo
```
git clone https://github.com/hmdliu/SRep -b final
cd SRep
```
The dataset will be automatically downloaded while training.

## Training script
1) On a HPC with a singlularity env and slurm
```
# remember to modify the path in the sbatch script
sbatch train.SBATCH [exp_id]
```
2) On a computer with GPU
```
# remember to activate the env
python train.py [exp_id]
```
Training log can be found in \[exp_id\].log

## Inference speed test
```
# On a HPC with a singlularity env and slurm
sbatch inference.SBATCH

# On a computer with GPU
python inference.py
```

## Util functions
```
# Check accuracy of multiple experiments
python helper.py dump

# Archive training logs
python helper.py log move
```