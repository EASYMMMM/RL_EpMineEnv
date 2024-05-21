@echo off
REM Activate the conda environment
call conda activate rl_robotcv

call python train.py train.algo=sac env.reward_scaling=true|| echo "First script failed, continuing..."

call python train.py train.algo=ppo env.reward_scaling=true

call python train.py train.algo=sac  

call python train.py train.algo=ppo  

REM Deactivate the conda environment (optional)
call conda deactivate
