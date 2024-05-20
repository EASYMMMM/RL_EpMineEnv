'''
python pendulum_train.py 
'''
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="cfg", config_name="EpMineEnvCfg.yaml")
def main(cfg : DictConfig) -> None:
    import os
    import gym
    import numpy as np
    import time
    from datetime import datetime
    from stable_baselines3 import PPO,DDPG,SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.utils import set_random_seed
    from envs.SingleAgent.mine_toy import EpMineEnv
    
    print(OmegaConf.to_yaml(cfg)) # 打印配置

    # 随机种子
    seed = cfg.train.seed

    env_id = cfg.env.env_id
    num_cpu = cfg.env.env_num  
    algo = {
        "sac": SAC,
        "ddpg": DDPG,
        "ppo": PPO,
    }[cfg.train.algo]
    n_timesteps = cfg.train.n_timesteps
    exp_name = cfg.env.exp_name   

    # dump config dict
    experiment_dir = os.path.join('runs', exp_name + 
    '_{date:%d-%H-%M-%S}'.format(date=datetime.now()))
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
        
    # 存放在sb3model/文件夹下
    save_path = experiment_dir + f"/{exp_name}_{algo}"

    # tensorboard log 路径
    tensorboard_log_path = experiment_dir+'/tensorboard_log'
    tensorboard_log_name = f"{exp_name}_{algo}"


    env_kwargs = {  "only_image":cfg.env.only_image,
                    "only_state":cfg.env.only_state,
                    "max_episode_steps":cfg.env.max_episode_steps}
                   
    # Instantiate and wrap the environment
    env = make_vec_env(env_id,
                       n_envs=num_cpu, 
                       seed=cfg.train.seed, 
                       vec_env_cls=DummyVecEnv,
                       env_kwargs = env_kwargs)

    hyperparams = {
        "sac": dict(
            batch_size=cfg.train.batch_size,
            gamma=cfg.train.gamma,
        ),
        "ddpg": dict(
            batch_size=cfg.train.batch_size,
            gamma=cfg.train.gamma,
        ),
        "ppo": dict(
            batch_size=cfg.train.batch_size,
            learning_rate=cfg.train.learning_rate,
            gamma=cfg.train.gamma,
            device=cfg.train.device,
            use_sde = cfg.train.use_sde,
            ent_coef = cfg.train.ent_coef
        )
    }[cfg.train.algo]

    begin_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    model = algo("CnnPolicy", env, verbose=1, tensorboard_log = tensorboard_log_path, **hyperparams,seed = seed)
    try:
        model.learn(n_timesteps , tb_log_name = tensorboard_log_name )
    except KeyboardInterrupt:
        pass
    print('=====================================')
    print(f"Saving to {save_path}.zip")
    model.save(save_path)
    end_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    print('Started at: ' + begin_time)
    print('Ended at: ' + end_time)
    print('=====================================')



if __name__ == '__main__':
    main()