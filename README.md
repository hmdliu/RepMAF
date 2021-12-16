## RepMAF: When Re-parameterization Meets Multi-scale Attention

#### Setup
**Requisites:** numpy, torch, torchvision, torchsummary.
Clone the project repo
```
git clone https://github.com/hmdliu/SRep -b final
cd SRep
```
The dataset will be automatically downloaded while training.

#### Training script
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

