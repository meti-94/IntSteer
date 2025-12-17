#!/bin/sh

#SBATCH --gres=gpu:1          # request 2 GPUs
#SBATCH --time=0
/bin/hostname

# run python job native on a node
srun --nodelist=f1-p1s27 --export OPENAI_API_KEY='sk-...' /home/unsw.mahdi/anaconda3/envs/agent_tom/bin/python \
     /home/unsw.mahdi/SAE-TS/src/sae_ts/baselines/analysis.py
