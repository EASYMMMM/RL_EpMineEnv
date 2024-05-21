import os
import os
import gym
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO,DDPG,SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from envs.SingleAgent.mine_toy import EpMineEnv
from cnnlstm_policy import CnnLSTMPolicy
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, List, Optional, Sequence, Type, Union
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn

env_kwargs = {  "only_image":True,
                "only_state":False,
                "reward_scaling":True,
                'dist_reward':'v1',
                'no_graph':False}

# 设置环境和模型路径
env_id = "EpMineEnv-v0"
num_cpu = 4
env = make_vec_env(env_id,
                    n_envs=num_cpu, 
                    seed=2, 
                    vec_env_cls=DummyVecEnv,
                    env_kwargs = env_kwargs) 
model_path = "runs\\RobotCv_ppo_21-21-47-27\\RobotCv_ppo.zip"
log_dir = "runs\\RobotCv_ppo_21-21-47-27\\tensorboard_log"
save_path = "runs\\RobotCv_ppo_21-21-47-27\\RobotCv_ppo_2.zip"
# 加载之前训练的模型
model = PPO.load(model_path, env=env, tensorboard_log=log_dir,)






# 继续训练
model.learn(total_timesteps=1000000, tb_log_name='RobotCv_ppo_2', reset_num_timesteps=False)
print('=====================================')
print(f"Saving to {save_path}.zip")
model.save(save_path)
print('=====================================')