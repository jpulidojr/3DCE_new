#/bin/bash

#To Run: salloc -N 1 --time=10:00:00 -p shared-gpu --constraint="gpu1_model:TITAN_V"
#Also:   salloc -N 1 --time=10:00:00 -p shared-gpu --constraint="gpu1_model:Quadro_P6000"

module unload python
module load anaconda/Anaconda2
module load cuda/9.0
conda activate CADLab
conda activate py2

python --version
