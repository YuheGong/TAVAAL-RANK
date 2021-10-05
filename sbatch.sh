#!/bin/bash
#SBATCH -p gpu_8
# #SBATCH -A
#SBATCH -J tavaal

# Please use the complete path details :
#SBATCH -D ./
#SBATCH -o ./slurm/out_%A_%a.log
#SBATCH -e ./slurm/err_%A_%a.log

# Cluster Settings
#SBATCH -n 1         # Number of tasks
#SBATCH -c 1  # Number of cores per task
#SBATCH -t 15:0:00             # 1:00:00 Hours, minutes and seconds, or '#SBATCH -t 10' - only minutes

#SBATCH --gres gpu:1
# -------------------------------

# Activate the virtualenv / conda environment

# Export Pythonpath


# Additional Instructions from CONFIG.yml


python main.py -m TA-VAAL -d cifar10 -c 5 $SLURM_ARRAY_TASK_ID

# THIS WAS BUILT FROM THE DEFAULLT SBATCH TEMPLATE