
import os
from getpass import getuser
import time
import argparse
import gym
import envs
import numpy as np
from stable_baselines3 import SAC, TD3, PPO
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Enjoy an RL agent trained using Stable Baselines3"
    ) 
    parser.add_argument(
        "--model",
        help="model name",
        type=str,
    )
    parser.add_argument(
        "--algo",
        help="algo",
        type=str,
    )
    parser.add_argument(
        "--save_path",
        help="save_path",
        type=str,
    )

    args = parser.parse_args()

    # 随机种子
    seed = 2

    # 环境名
    env_id = "EpMineEnv-v0"

    # env_kwargs = {  "only_image":cfg.env.only_image,
    #                 "only_state":cfg.env.only_state,
    #                 "max_episode_steps":cfg.env.max_episode_steps}
    
    # Create an env similar to the training env
    env = gym.make(env_id) 
    # env = gym.make(env_id,**env_kwargs)
    
    algo = {
        "sac": SAC,
        "td3": TD3,
        "ppo": PPO,
    }[args.algo]


    # save_path = 'runs/'+args.model+'/RobotCv_ppo.zip'
    save_path = args.save_path
    # save_path = 'runs/'+args.model+'/RobotCv_sac.zip'
    print('load from:')

    print(save_path)
    # Load the saved model
    model = algo.load(save_path, env=env)


    print("==============================")
    # print(f"gradient steps:{model.gradient_steps}")
    print("model path:"+save_path)
    print("==============================")

    episode_rewards, episode_lengths, episode_ave_velocitys, episode_success_rate = [], [], [], []
    for __ in range(5):
        obs = env.reset()
        episode_reward = 0.0
        episode_length = 0
        for _ in range(1750):
            #time.sleep(0.02)
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            print('robot position:',info['robot_position'])
            # print('catch state:',info['catch_state'])
            episode_reward += rewards
            episode_length += 1
            if dones:
                is_success  = True
                episode_success_rate.append(is_success)
                break   
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
            
        print(
            f"Episode {len(episode_rewards)} reward={episode_reward}, length={episode_length}"
        )
        print('success:',is_success)
        print('************************')

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    mean_len, std_len = np.mean(episode_lengths), np.std(episode_lengths)

    success_rate = sum(episode_success_rate)/len(episode_success_rate)
    print("========== Results ===========")
    print(f"Episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Episode_length={mean_len:.2f} +/- {std_len:.2f}")
    print(f"Episode_success_rate={success_rate:.2f}")
    print("==============================")

    # Close process
    env.close()