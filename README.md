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
安装`hydra`和`Omegacfg`来配置实验参数。
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

检查训练结果：
```
python play.py --model RobotCv_20-22-21-06
```

## 3. RL Env设置
### 3.1 观测空间
cv：机器人视觉信号
state：机器人信息[ 四元数 速度 全局位置 ...]
原1 采用State 
原2 采用cv
修改后 cv信号+机器人速度+朝向（不给全局位置）

### 3.2 动作空间


### 3.3 奖励函数
原1 稀疏奖励函数（任务成功+10）
原2 稠密奖励函数（稀疏+相对距离）
修改后 xxx奖励函数

### 3.4 强化学习算法
PPO SAC

## 4. 实验设置及结果

### 4.1 Baseline
PPO + 原稠密奖励函数 + 视觉

### 4.2 提出的方法
SAC + 修改后奖励函数 + 修改后状态空间 + trick

### 4.n 消融实验
