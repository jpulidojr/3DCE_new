#/bin/bash

module unload python
module load anaconda/Anaconda2
module load cuda/9.0
conda activate CADLab
conda activate py2

python --version
