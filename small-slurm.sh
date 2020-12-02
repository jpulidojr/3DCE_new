#! /bin/bash
#SBATCH --partition=shared-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu1_model:Quadro_P6000
#SBATCH --time=00:20:00
date

cd /home/pulido/CADLab/lesion_detector_3DCE/3DCE_new
source loadModules.sh

make -j

./train.sh

date
