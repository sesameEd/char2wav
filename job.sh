#!/bin/bash
#SBATCH --account=def-chael
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=4096M               # memory (per node)
#SBATCH --time=0-08:00            # time (DD-HH:MM)
#SBATCH --job-name=train_c2v
#SBATCH --output=data/%x-%j.out
# source setup.sh
./train_char2voc.py --ss -E 30 --no_voice --init
# ./train_char2voc.py --frac 84 -E 3 -B 4 -T 2 --init --no_voice
# tensorboard --host 0.0.0.0 --logdir=data/tensorboard/
# ssh -L 16006:localhost:6006 cedar