
import os
from getpass import getuser
import time
import argparse
import gym
import envs
import numpy as np
from stable_baselines3 import SAC, TD3, PPO
import matplotlib.pyplot as plt
import os
import imageio
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def make_gif(obs_sequence):
    # Select every nth frame to reduce the total number of frames
    nth_frame = 5  # Change this value to skip more frames if needed
    selected_frames = obs_sequence[::nth_frame]

    # Create and save a GIF
    frames = []
    i=1
    for frame in selected_frames:
        print(i)
        i=i+1
        fig, ax = plt.subplots()
        ax.imshow(frame)
        ax.axis('off')
        fig.canvas.draw()
        
        # Convert the plot to a numpy array
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        frames.append(image)
        plt.close(fig)

    # Save frames as a GIF
    imageio.mimsave('robot_navigation.gif', frames, fps=10)

    print("GIF created and saved as 'robot_navigation.gif'")

def plot_robot_trajectory(positions, map_size=5, step=2, savepath='trajectory_result.jpg'):
    """
    Function to plot the robot's trajectory.
    :param positions: List of robot positions (each position is a list of 3 coordinates).
    :param map_size: Size of the map (default is 2.5x2.5).
    :param step: Step interval for plotting points to reduce clutter.
    """
    if not positions:
        print("No positions to plot.")
        return
    
    plt.figure(figsize=(8, 8))
    
    # Extract x and z coordinates (2D plane) every 'step' points
    x_coords = [pos[0] for i, pos in enumerate(positions) if pos is not None and i % step == 0]
    z_coords = [pos[2] for i, pos in enumerate(positions) if pos is not None and i % step == 0]
    
    if not x_coords or not z_coords:
        print("Position data is empty.")
        return

    # Normalize the color values between 0 and 1 based on the sequence
    colors = np.linspace(0, 1, len(x_coords))

    # Create a line plot with a colormap
    points = np.array([x_coords, z_coords]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    from matplotlib.collections import LineCollection
    lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(0, 1))
    lc.set_array(colors)
    lc.set_linewidth(2)
    
    plt.gca().add_collection(lc)
    plt.colorbar(lc, label='Progression of Time')  # Add a color bar to show the progression
    plt.scatter(0, 0, color='red', s=100)  # Add a red point at the origin
    plt.xlim(-map_size/2, map_size/2)
    plt.ylim(-map_size/2, map_size/2)
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    plt.title("Robot Trajectory")
    plt.xlabel("X Coordinate")
    plt.ylabel("Z Coordinate")
    plt.grid(True)
    plt.savefig(savepath, bbox_inches='tight')
    plt.show()
    

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
        default='ppo'
    )
    parser.add_argument(
        "--save_path",
        help="save_path",
        type=str,
    )

    args = parser.parse_args()

    # 随机种子
    seed = 1

    # 环境名
    env_id = "EpMineEnv-v0"

    env_kwargs = {   "reward_scaling": True,
                    'dist_reward':'v1',}
                     
    
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
    model = algo.load(save_path, 
                      # seed=seed,
                       env=env,)


    print("==============================")
    # print(f"gradient steps:{model.gradient_steps}")
    print("model path:"+save_path)
    print("==============================")
    print('model policy:')
    # 打印模型的网络结构
    print(model.policy)
    print("==============================")

    episode_rewards, episode_lengths, episode_ave_velocitys, episode_success_rate = [], [], [], []
    all_robot_positions = []
    obs = env.reset()
    #time.sleep(3)
    for __ in range(1):
        obs_sequence = [ ]
        obs = env.reset()
        episode_reward = 0.0
        episode_length = 0
        robot_positions = []  # To store positions for each episode
        is_success = False
        for _ in range(950):
            # time.sleep(0.02)
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            obs_sequence.append(obs)
            robot_positions.append(info['robot_position'])
            # print('robot position:',info['robot_position'])
            # print('robot rotation', info["robot_rotation"])
            # print('catch state:',info['catch_state'])
            episode_reward += rewards
            episode_length += 1
            if dones:
                is_success  = True
                
                break   
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        all_robot_positions.append(robot_positions)
        episode_success_rate.append(is_success)    
        print(
            f"Episode {len(episode_rewards)} reward={episode_reward}, length={episode_length}"
        )
        print('success:',is_success)
        print('************************')

    plot_robot_trajectory(all_robot_positions[0])

    obs_sequence = np.array(obs_sequence)
    make_gif(obs_sequence)

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

