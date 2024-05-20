# 强化学习 机器人视觉导航
## 1. 任务描述
在固定环境内，根据第一视角图像输入，找到指定目标。

状态：第一视角图像，尺寸为128x128

动作：机器人横向速度、纵向速度和旋转角速度

奖励：机器人到达指定位置会返回+10奖励，在`envs/SingleAgent/mine_toy.py`中设置了一种简易稠密奖励方式。

对环境的测试可以参考`envs/SingleAgent/mine_toy.py`文件。

## 2. 环境配置

### 使用本地环境（有显示器）

```
conda create -p python3.8 mine_env
conda activate mine_env
pip install mlagents-envs gym opencv-python==4.5.5.64 stablebaselin3==1.5.0
```

根据系统配置和[官方文档](https://pytorch.org/get-started/locally/)安装pytorch。

~~根据[Stable-Baseline3](https://stable-baselines3.readthedocs.io/en/master/guide/install.html)官方文档安装强化学习框架。~~

### 使用远程服务器（无显示器）

为了使仿真环境支持服务器端的图像渲染，建议使用docker的方式。需要安装`nvidia-docker`，详细的安装过程建议参考[https://github.com/ValerioB88/ml-agents-visual-observations-docker-GPU](https://github.com/ValerioB88/ml-agents-visual-observations-docker-GPU)。

我们也提供了已经配置好环境的镜像，上述过程配置完成后，可以直接拉取镜像

```
docker pull haoranlee/mine_visual_gpu:v0
```

使用该镜像时，可以将本地的代码和仿真环境所在文件夹映射到docker容器中
```
docker run -it -e DISPLAY=:0 -v /your/workspace/path:/work/code --network host --runtime=nvidia --privileged --entrypoint /bin/bash haoranlee/mine_visual_gpu:v0
```

需要注意的是，在有显示器的机器上使用上述docker方式，仍然会弹出程序可视化窗口。


### 关闭可视化界面
mlagents-envs提供了`no-graphics`仿真模式，但是在该模式下图像不会被正常渲染。
这里我们提供了一种通过修改mlagents-envs源码的方式，让它们支持不显示可视化窗口。
具体的，找到当前python环境的库安装路径，并找到`site-packages/mlagents_envs/environment.py`，将第272行
`args += ["-nographcis", "-batchmode"]` 修改为 `args += ["-batchmode"]`。

然后再代码（`envs/SingleAgent/mine_toy.py`）中 `no_graph = True`。

需要注意的是，上述修改方式虽然支持关闭可视化窗口，但是在服务器（无显示）端仅修改上述代码而不适用docker的情况下，仍然不能正常渲染图像。

***警告***：上述代码涉及修改mlagents-envs源码，请谨慎使用。

## 仿真环境下载
在[release](https://github.com/DRL-CASIA/EpMineEnv/releases)标签下，下载最新的系统对应的仿真环境，解压到`envs/SingleAgent/`路径下，并检查`envs/SingleAgent/mine_toy.py`中的`file_name`路径是否正确。

在Linux系统下，需要赋予仿真环境`drl.x86_64`文件可执行权限，具体如下
```
chmod +x drl.x86_64
```

### 训练
安装'hydra'和'Omegacfg'来配置实验参数。
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
