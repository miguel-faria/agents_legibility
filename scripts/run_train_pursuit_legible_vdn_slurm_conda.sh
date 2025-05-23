#!/bin/bash

#SBATCH --mail-type=BEGIN,END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=miguel.faria@tecnico.ulisboa.pt
#SBATCH --job-name=train_pursuit_legible_vdn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
# #SBATCH --gres=gpu:1
#SBATCH --gres=gpu:quadro6000:1
#SBATCH --time=7-00:00:00
#SBATCH --mem=4G
#SBATCH --qos=gpu-long
#SBATCH --output="job-%x-%j.out"
# #SBATCH --partition=a6000
date;hostname;pwd

if [ -n "${SLURM_JOB_ID:-}" ] ; then
  script_path=$(dirname "$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}' | head -n 1)")
else
  script_path="$( cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 ; pwd -P )"
fi

#export LD_LIBRARY_PATH="/opt/cuda/lib64:$LD_LIBRARY_PATH"
#export PATH="/opt/cuda/bin:$PATH"

#module load python cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
if [ "$HOSTNAME" = "artemis" ] || [ "$HOSTNAME" = "poseidon" ] ; then
  if [ -z "$CONDA_PREFIX_1" ] ; then
    conda_dir="$CONDA_PREFIX"
  else
    conda_dir="$CONDA_PREFIX_1"
  fi
else
  conda_dir="$CONDA_HOME"
fi

source "$conda_dir"/bin/activate drl_env
if [ "$HOSTNAME" = "artemis" ] || [ "$HOSTNAME" = "poseidon" ] || [ "$HOSTNAME" = "maia" ]  ; then
  python "$script_path"/run_train_pursuit_legible_vdn_dqn.py --field-len 10 --hunters 2 --catch-reward 5 --prey-type random --batch-size 64 --buffer-size 30000 --iterations 100000 --episode-steps 800 --warmup 1600 --limits 1 1 --eps-type linear --eps-decay 0.5 --start-eps 0.75 --legible-reward info --beta 0.9 --legibility-temp 0.1 --online-lr 0.00005 --train-thresh 0.96 --use-lower-curriculum --logs-dir /mnt/scratch-artemis/miguelfaria/logs/pursuit --models-dir /mnt/data-artemis/miguelfaria/deep_rl/models --data-dir /mnt/data-artemis/miguelfaria/deep_rl/data
elif [ "$HOSTNAME" = "hera" ] ; then
  nohup python "$script_path"/run_train_pursuit_legible_vdn_dqn.py --field-len 10 --hunters 2 --prey-type idle --catch-reward 5 --batch-size 64 --buffer-size 5000 --iterations 15000 --episode-steps 1000 --warmup 1000 --limits 1 4 --eps-type linear --eps-decay 0.27 --legible-reward info --beta 0.8 --legibility-temp 0.1 --online-lr 0.00005 --use-higher-curriculum --logs-dir /mnt/data-hera1/miguelfaria/deep_rl/logs/pursuit --models-dir /mnt/data-hera1/miguelfaria/deep_rl/models --data-dir /mnt/data-hera1/miguelfaria/deep_rl/data
else
  python "$script_path"/run_train_pursuit_legible_vdn_dqn.py --field-len 10 --hunters 2 --catch-reward 5 --prey-type random --batch-size 64 --buffer-size 5000 --iterations 50000 --episode-steps 800 --warmup 800 --limits 1 1 --eps-type linear --eps-decay 0.05 --start-eps 0.75 --legible-reward info --beta 0.95 --legibility-temp 0.1 --online-lr 0.0005 --train-thresh 0.96 --use-lower-curriculum --models-dir /mnt/d/Research/models/legibility
fi

conda deactivate
date
