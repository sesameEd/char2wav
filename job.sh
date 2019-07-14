#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --account=sesame
./train_char2voc.py --ss --init -E 20