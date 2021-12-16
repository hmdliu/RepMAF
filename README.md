## RepMAF: When Re-parameterization Meets Multi-scale Attention

CSCI-GA 2271: Computer Vision - Final Project (Tutored by Prof. Rub Fergus)

## Setup
**Requisites:** numpy, torch, torchvision, torchsummary. \
Clone the project repo:
```
git clone https://github.com/hmdliu/SRep -b final
cd SRep
```
The dataset will be automatically downloaded while training.

## Training script
1) On a HPC with a singlularity env and slurm:
```
# remember to modify the path in the sbatch script
sbatch train.SBATCH [exp_id]
```
2) On a computer with GPU:
```
# remember to activate the env
python train.py [exp_id]
```
Training log can be found in \[exp_id\].log

## Experiment IDs
The IDs follow the original order in the report. \
\
**Table 1**: vgg-idt, vgg-se, repvgg-idt, repvgg-se,birepvgg-idt3, birepvgg-se3, repmaf-maf3, repmaf-maf4. \
\
**Table 2**: repvgg-idt, repvgg-ses, repvgg-se. \
(**Note:** To disable data augmentation, please set config\['aug'\] = False in *config.py*.) \
\
**Table 3**: repvgg-se3, repvgg-se2, repvgg-se1, repvgg-se3, repmaf-maf5, repmaf-maf3, repmaf-maf1, repmaf-maf6, repmaf-maf4, repmaf-maf2. 

## Inference speed test
```
# On a HPC with a singlularity env and slurm
sbatch inference.SBATCH

# On a computer with GPU
python inference.py
```

## Util functions
```
# Check best pred of multiple experiments
python helper.py dump

# Archive training logs
python helper.py log move
```