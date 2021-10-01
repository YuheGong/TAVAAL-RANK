# Cluster Settings
#SBATCH -n 1         # Number of tasks
#SBATCH -c 1  # Number of cores per task
#SBATCH -t 2:0:00             # 1:00:00 Hours, minutes and seconds, or '#SBATCH -t 10' - only minutes

#SBATCH --gres gpu:1
# -------------------------------

# Activate the virtualenv / conda environment



# Export Pythonpath

python main.py -m TA-VAAL -d cifar10 -c 5 $SLURM_ARRAY_TASK_ID

# THIS WAS BUILT FROM THE DEFAULLT SBATCH TEMPLATE
