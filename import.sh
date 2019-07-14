#!/bin/bash
#SBATCH --account=def-chael
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=100M               # memory (per node)
#SBATCH --time=0-08:00            # time (DD-HH:MM)
#SBATCH --job-name=install_sf
#SBATCH --output=%x-%j.out
python3 -c "import soundfile as sf"
# conda activate
# pip install --user --global-option="-I/home/sesame/.local/include" \
    # --global-option=build_ext --global-option="-L/home/sesame/.local/lib" sndfile