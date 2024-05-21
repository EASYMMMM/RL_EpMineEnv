# 强化学习 机器人视觉导航
## 1. 任务描述
在固定环境内，根据第一视角图像输入，找到指定目标。

状态：第一视角图像，尺寸为128x128

动作：机器人横向速度、纵向速度和旋转角速度

奖励：机器人到达指定位置会返回+10奖励，在`envs/SingleAgent/mine_toy.py`中设置了一种简易稠密奖励方式。

对环境的测试可以参考`envs/SingleAgent/mine_toy.py`文件。

## 2. 环境配置



```
conda create -p python3.8 mine_env
conda activate mine_env
pip install mlagents-envs gym opencv-python==4.5.5.64 stablebaselin3==1.5.0
```

根据系统配置和[官方文档](https://pytorch.org/get-started/locally/)安装pytorch。


### 关闭可视化界面
mlagents-envs提供了`no-graphics`仿真模式，但是在该模式下图像不会被正常渲染。
这里我们提供了一种通过修改mlagents-envs源码的方式，让它们支持不显示可视化窗口。
具体的，找到当前python环境的库安装路径，并找到`site-packages/mlagents_envs/environment.py`，将第272行
`args += ["-nographcis", "-batchmode"]` 修改为 `args += ["-batchmode"]`。

然后再代码（`envs/SingleAgent/mine_toy.py`）中 `no_graph = True`。

需要注意的是，上述修改方式虽然支持关闭可视化窗口，但是在服务器（无显示）端仅修改上述代码而不适用docker的情况下，仍然不能正常渲染图像。

***警告***：上述代码涉及修改mlagents-envs源码，请谨慎使用。


### 训练
安装'hydra'和'Omegacfg'来配置实验参数。
``` 
pip install hydra-core --upgrade
```
原训练：
```
python train_ppo.py
```
默认参数配置在`cfg/EpMineEnvCfg.yaml`中。
使用默认参数训练（PPO baseline）：
```
python train.py 
```
可在命令行中修改`.yaml`中已有的参数：
```
python train.py env.only_image=False train.algo=sac train.n_timesteps=1500000
```
训练结果、配置文件、模型等均会保存在`\runs`路径下。

检查训练结果，需要指定使用的算法，给出模型路径：
```
python play.py --algo ppo --save_path runs\\RobotCv_ppo_21-12-34-08\\RobotCv_ppo.zip
```

## 3. RL Env设置
### 3.1 观测空间
cv：机器人视觉信号
state：机器人信息[ 四元数 速度 全局位置 ...]
原1 采用State 
原2 采用cv
~~修改后 cv信号+机器人速度+朝向（不给全局位置）~~
我们按照作业要求，仅用机器人视觉信号作为观测输入。

### 3.2 动作空间


### 3.3 奖励函数
原 稀疏奖励函数（任务成功+10） 未使用
原 稠密奖励函数（稀疏+相对距离）
在我们的项目中，定义了两个稠密奖励：
- v0: 即原稠密奖励函数。计算机器人到目标矿石的距离，并进一步计算机器人到目标矿石的移动速度。以速度作为奖励。
```python
    def get_dist_reward_v0(self,results):
        # 到目标的距离奖励
        current_dist = self.get_dist_to_mine(reuslts=results)
        # print(self.last_dist - current_dist)
        dist_reward = (self.last_dist - current_dist) 
        self.last_dist = current_dist
        return dist_reward
```
由于速度值较小，该奖励值也较小（0.000x的量级）。为了提高训练稳定性，我们希望对奖励进行缩放（reward scaling），减小其方差。然而该速度值太小，经尝试，缩放效果不好。

于是，设计新的奖励函数，惩罚机器人到矿石的距离，不再计算速度。并进行缩放。
- v1:
```python
    def get_dist_reward_v1(self,results):
        # 负距离表示
        current_dist = self.get_dist_to_mine(reuslts=results)
        # print(self.last_dist - current_dist)
        dist_reward = (- current_dist) 
        self.last_dist = current_dist

        if self.reward_scaling:
            dist_reward = (dist_reward+3)/3
        return dist_reward
```
该奖励函数可在配置文件`.yaml`中更改。

### 3.4 强化学习算法
PPO SAC

### 3.5 网络设置
给出两种Policy：
- CnnPolicy: 这是stable-baselines3给出的带有卷积网络的默认策略。其结构为：
    ```
    ActorCriticCnnPolicy(
    (features_extractor): NatureCNN(
        (cnn): Sequential(
        (0): Conv2d(3, 32, kernel_size=(8, 8), stride=(4, 4))
        (1): ReLU()
        (2): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
        (3): ReLU()
        (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        (5): ReLU()
        (6): Flatten(start_dim=1, end_dim=-1)
        )
        (linear): Sequential(
        (0): Linear(in_features=9216, out_features=512, bias=True)
        (1): ReLU()
        )
    )
    (mlp_extractor): MlpExtractor(
        (shared_net): Sequential()
        (policy_net): Sequential()
        (value_net): Sequential()
    )
    (action_net): Linear(in_features=512, out_features=3, bias=True)
    (value_net): Linear(in_features=512, out_features=1, bias=True)
    )
    ```
然而，由于观测空间仅有视觉输入，缺乏机器人速度、加速度、全局位置等信息，导致任务信息缺失（POMDP）。我们使用LSTM处理特征，从而提供更全面的时序信息。
- CnnLstmPolicy: 定义在`cnnlstm_policy.py`中。

## 4. 实验设置及结果

### 4.1 Exp0：Baseline
PPO + 奖励函数v0 + CnnPolicy

### 4.2 Exp1：更改奖励函数

- **EXP1:** PPO + 修改后奖励函数v1 + CnnPolicy
  ```
  python train.py train.algo=ppo env.reward_scaling=true env.dist_reward=v1 
  ```
- **EXP1.1:** PPO + 修改后奖励函数v1 + CnnPolicy + ent_coef=0.1
  ```
  python train.py train.algo=ppo env.reward_scaling=true env.dist_reward=v1 train.ent_coef=0.1
  ```
  ent_coef表示loss中的熵系数，用于鼓励模型探索。然而，相比于EXP1，EXP1.1在训练后期可能由于过度探索而导致训练严重不稳定。

(可能还有Exp1.2，添加target_kl参数来避免曲线下跌)
```
python train.py train.algo=ppo env.reward_scaling=true env.dist_reward=v1 train.target_kl=1.5 env.no_graph=True
```

### 4.3 Exp2: LSTM
PPO + 修改后奖励函数v1 + CnnLstmPolicy
```
python train.py train.algo=ppo env.reward_scaling=true env.dist_reward=v1 train.policy=CnnLstmPolicy
```
(还没跑，可能还有2.1，使用奖励函数v0测试一下)

