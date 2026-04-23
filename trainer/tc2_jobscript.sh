#!/bin/bash

### TC2 Job Script ###
 
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1

### Specify Memory allocate to this job ###
#SBATCH --mem=30G

### Specify number of core (CPU) to allocate to per task ###
#SBATCH --cpus-per-task=10

### Specify number of node to compute ###
#SBATCH --nodes=1

### Optional: Specify node to execute the job ###
### Remove 1st # at next line for the option to take effect ###
##SBATCH --nodelist=TC2N01

### Specify Time Limit, format: <min> or <min>:<sec> or <hr>:<min>:<sec> or <days>-<hr>:<min>:<sec> or <days>-<hr> ### 
#SBATCH --time=06:00:00

### Specify name for the job, filename format for output and error ###

#SBATCH --job-name=MOE_gating
#SBATCH --output=../logs/%x_%j/output.out
#SBATCH --error=../logs/%x_%j/error.err

### Your script for computation ###
module load anaconda
eval "$(conda shell.bash hook)"

#conda create -n AI6130_Project python=3.9 -y
conda activate AI6130_Project

#conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
#conda install -y -c conda-forge sentencepiece protobuf transformers tokenizers datasets accelerate evaluate

# wget -P ./dataset/data https://www.modelscope.cn/datasets/gongjy/minimind_dataset/resolve/master/pretrain_t2t_mini.jsonl
# wget -P ./dataset/data https://www.modelscope.cn/datasets/gongjy/minimind_dataset/resolve/master/sft_t2t_mini.jsonl



python ../dataset/data/sft_split.py

python train_pretrain.py --data_path ../dataset/data/pretrain_t2t_mini.jsonl --use_moe 1 --batch_size 128 --num_batches 5000
python train_full_sft.py --data_path ../dataset/data/sft_train.jsonl --use_moe 1 --batch_size 64 --num_batches 5000

cd ../

python ./experiments.py
python ./load_model.py


conda deactivate
#conda env remove --name AI6130_Assignment2_C250113 -y